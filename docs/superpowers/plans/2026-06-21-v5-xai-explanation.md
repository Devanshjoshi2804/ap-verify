# Unified XAI / explanation layer (v5 slice 5) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A unified, ranked, structured `Explanation` behind any fraud decision — exact linear attribution for the fusion model and glass-box condition attribution for the rule detectors — surfaced on the live pipeline and a demo CLI.

**Architecture:** `Explanation`/`Factor` types and the rule-detector builders live in pure domain; `explain_fusion` lives in `eval/` (where the linear model is) and imports the domain types; the pipeline derives explanations for live flags. No new dependencies.

**Tech Stack:** Python 3.12, stdlib, frozen dataclasses, typer/rich, pytest.

**Spec:** `docs/superpowers/specs/2026-06-21-v5-xai-explanation-design.md`

## Global Constraints

- Clean/hexagonal: `domain` imports only stdlib + `domain`; `eval`/`application` import inward.
- No new dependencies; no `shap`. Fusion attribution is exact closed-form `weight × feature`.
- `Factor.contribution` is signed; factors in an `Explanation` are sorted by `|contribution|` descending.
- Determinism: no randomness.
- Gates after every task: `ruff check .`, `ruff format --check .`, `mypy --strict src tests`, `pytest`. **Domain layer 100% coverage.**
- No AI-tells; match surrounding idiom.
- **Git note:** not a git repo, so `git commit` steps are no-ops; the per-task checkpoint is the gate suite.

---

### Task 1: `Explanation`/`Factor` + rule-detector builders (domain)

**Files:**
- Create: `src/apverify/domain/explanation.py`
- Test: `tests/unit/domain/test_explanation.py`

**Interfaces:**
- Consumes: `apverify.domain.fraud.DuplicateMatch`, `apverify.domain.vendor_master.{VendorRiskAssessment, Severity}`, `apverify.domain.anomaly.{AnomalyAssessment, AnomalySeverity}`
- Produces:
  - `Factor(signal: str, value: str, contribution: float, detail: str)` (frozen)
  - `Explanation(source: str, headline: str, factors: tuple[Factor, ...])` (frozen)
  - `explanation(source: str, headline: str, factors: Sequence[Factor]) -> Explanation` (sorts factors by `|contribution|` desc)
  - `explain_duplicate(match: DuplicateMatch) -> Explanation`
  - `explain_vendor_risk(assessment: VendorRiskAssessment) -> Explanation`
  - `explain_anomaly(assessment: AnomalyAssessment) -> Explanation`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/domain/test_explanation.py`:

```python
from __future__ import annotations

from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.explanation import (
    Factor,
    explain_anomaly,
    explain_vendor_risk,
    explanation,
)
from apverify.domain.vendor_master import Severity, VendorRiskAssessment, VendorRiskKind


def test_factors_are_ranked_by_absolute_contribution() -> None:
    result = explanation(
        "test",
        "headline",
        [Factor("a", "1", 0.2, ""), Factor("b", "2", -0.9, ""), Factor("c", "3", 0.5, "")],
    )
    assert [factor.signal for factor in result.factors] == ["b", "c", "a"]


def test_vendor_risk_explanation_leads_with_the_kind() -> None:
    assessment = VendorRiskAssessment(
        VendorRiskKind.BANK_CHANGE, Severity.HIGH, 1.0, "ACME", "bank changed"
    )
    result = explain_vendor_risk(assessment)
    assert result.factors[0].value == "bank_change"
    assert "bank changed" in result.factors[0].detail


def test_anomaly_explanation_names_the_top_feature() -> None:
    assessment = AnomalyAssessment(0.95, AnomalySeverity.HIGH, "amount_spike", "11x median")
    result = explain_anomaly(assessment)
    assert result.factors[0].signal == "amount_spike"
    assert result.factors[0].contribution == 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/domain/test_explanation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.domain.explanation'`

- [ ] **Step 3: Implement `domain/explanation.py`**

```python
"""Unified, structured explanations behind a fraud decision.

Every detector emits a free-text reason; this turns each decision into a *ranked* set of
contributing factors that read the same way across sources. The rule detectors are
glass-box, so their factors enumerate the conditions that fired; the fusion model's
factors (built in ``eval``) are exact ``weight × feature`` linear contributions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.fraud import DuplicateMatch
from apverify.domain.vendor_master import Severity, VendorRiskAssessment

_SEVERITY_WEIGHT = {
    Severity.HIGH: 1.0,
    Severity.LOW: 0.3,
    Severity.NONE: 0.0,
    AnomalySeverity.HIGH: 1.0,
    AnomalySeverity.MEDIUM: 0.6,
    AnomalySeverity.NONE: 0.0,
}


@dataclass(frozen=True, slots=True)
class Factor:
    signal: str
    value: str
    contribution: float  # signed; positive = toward the flag (toward P(correct) for fusion)
    detail: str


@dataclass(frozen=True, slots=True)
class Explanation:
    source: str
    headline: str
    factors: tuple[Factor, ...]


def explanation(source: str, headline: str, factors: Sequence[Factor]) -> Explanation:
    ranked = tuple(sorted(factors, key=lambda factor: abs(factor.contribution), reverse=True))
    return Explanation(source=source, headline=headline, factors=ranked)


def explain_duplicate(match: DuplicateMatch) -> Explanation:
    return explanation(
        "duplicate",
        f"{match.tier.value} (score {match.score:.2f})",
        [
            Factor("duplicate_tier", match.tier.value, match.score, match.reason),
            Factor("matched_id", match.matched_id, match.score * 0.5, f"matched {match.matched_id}"),
        ],
    )


def explain_vendor_risk(assessment: VendorRiskAssessment) -> Explanation:
    return explanation(
        "vendor-master",
        f"{assessment.kind.value} ({assessment.severity.value})",
        [
            Factor(
                "vendor_risk",
                assessment.kind.value,
                _SEVERITY_WEIGHT[assessment.severity],
                assessment.reason,
            ),
            Factor(
                "name_similarity",
                f"{assessment.score:.2f}",
                assessment.score * 0.5,
                f"nearest known vendor: {assessment.matched_vendor}",
            ),
        ],
    )


def explain_anomaly(assessment: AnomalyAssessment) -> Explanation:
    return explanation(
        "anomaly",
        f"{assessment.top_feature} ({assessment.severity.value})",
        [Factor(assessment.top_feature, f"{assessment.score:.2f}", assessment.score, assessment.reason)],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/domain/test_explanation.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/domain/explanation.py tests/unit/domain/test_explanation.py
git commit -m "feat(xai): unified Explanation type + rule-detector builders"
```

---

### Task 2: `explain_fusion` — exact linear attribution (eval)

**Files:**
- Modify: `src/apverify/eval/fusion.py`
- Test: `tests/unit/eval/test_fusion.py` (append; create if absent)

**Interfaces:**
- Consumes: `apverify.domain.explanation.{Factor, Explanation, explanation}`, `FEATURES`, `LogisticRegression`, `FeatureRow` (same module)
- Produces: `explain_fusion(model: LogisticRegression, row: FeatureRow) -> Explanation`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/eval/test_fusion.py`:

```python
from apverify.eval.fusion import FeatureRow, LogisticRegression, explain_fusion


def _row(**over: object) -> FeatureRow:
    base: dict[str, object] = dict(
        label="d",
        field="total",
        critic_confidence=0.5,
        verbalized_confidence=0.5,
        cross_check_passed=True,
        arithmetic_passed=False,
        format_passed=True,
        cross_model_agrees=True,
        correct=False,
    )
    base.update(over)
    return FeatureRow(**base)  # type: ignore[arg-type]


def test_explain_fusion_contribution_is_weight_times_feature() -> None:
    # weights align to FEATURES: critic, verbalized, cross_check, arithmetic, format, cross_model.
    model = LogisticRegression(weights=(0.0, 0.0, 0.0, 2.0, 0.0, 0.0), bias=0.0)
    result = explain_fusion(model, _row(arithmetic_passed=False))
    arithmetic = next(f for f in result.factors if f.signal == "arithmetic_passed")
    assert arithmetic.contribution == 0.0  # 2.0 * 0.0 (failed)
    result_passed = explain_fusion(model, _row(arithmetic_passed=True))
    arithmetic_passed = next(f for f in result_passed.factors if f.signal == "arithmetic_passed")
    assert arithmetic_passed.contribution == 2.0  # 2.0 * 1.0


def test_explain_fusion_ranks_the_dominant_signal_first() -> None:
    model = LogisticRegression(weights=(0.1, 0.0, 0.0, 3.0, 0.0, 0.0), bias=0.0)
    result = explain_fusion(model, _row(arithmetic_passed=True, critic_confidence=1.0))
    assert result.factors[0].signal == "arithmetic_passed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/eval/test_fusion.py -k explain_fusion -v`
Expected: FAIL with `ImportError: cannot import name 'explain_fusion'`

- [ ] **Step 3: Implement**

Add to `src/apverify/eval/fusion.py` the import (with the other imports):

```python
from apverify.domain.explanation import Explanation, Factor, explanation
```

And the function (place after `evaluate_fusion`):

```python
def explain_fusion(model: LogisticRegression, row: FeatureRow) -> Explanation:
    """Exact linear attribution: each signal's contribution to the log-odds of *correct*
    is ``weight × feature`` — SHAP for a linear model, in closed form. Positive raises
    P(correct); negative lowers it."""
    values = row.features()
    factors = [
        Factor(
            signal=name,
            value=f"{value:.2f}",
            contribution=weight * value,
            detail=f"weight {weight:+.2f} × {value:.2f}",
        )
        for name, weight, value in zip(FEATURES, model.weights, values, strict=True)
    ]
    factors.append(Factor("bias", "", model.bias, "model baseline"))
    probability = model.predict_proba(values)
    headline = f"P(correct) {probability:.2f}"
    return explanation("fusion", headline, factors)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/eval/test_fusion.py -k explain_fusion -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/fusion.py tests/unit/eval/test_fusion.py
git commit -m "feat(xai): exact linear attribution for the fusion model"
```

---

### Task 3: Derive explanations on `PayableReview`

**Files:**
- Modify: `src/apverify/application/review_payable.py`
- Test: `tests/unit/application/test_review_payable_anomaly.py` (append)

**Interfaces:**
- Consumes: `apverify.domain.explanation.{Explanation, explain_vendor_risk, explain_anomaly}`, `apverify.domain.vendor_master.Severity`, `apverify.domain.anomaly.AnomalySeverity`
- Produces: `PayableReview.explanations: tuple[Explanation, ...]`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/application/test_review_payable_anomaly.py`:

```python
def test_flagged_invoice_carries_a_structured_explanation() -> None:
    review = _use_case(_history([9000, 9100, 9200, 9300, 9150])).execute(_DOC)
    assert review.explanations  # at least one explanation for the anomaly flag
    assert any(e.source == "anomaly" for e in review.explanations)


def test_clean_invoice_has_no_explanations() -> None:
    review = _use_case(_history([180000, 184000, 184200, 185000, 188000])).execute(_DOC)
    assert review.explanations == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/application/test_review_payable_anomaly.py -k explanation -v`
Expected: FAIL with `AttributeError: 'PayableReview' object has no attribute 'explanations'`

- [ ] **Step 3: Implement**

In `src/apverify/application/review_payable.py`:

Add the domain imports:

```python
from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.explanation import Explanation, explain_anomaly, explain_vendor_risk
from apverify.domain.vendor_master import Severity, VendorRiskAssessment, assess_vendor_risk
```

(`AnomalyAssessment`, `Severity`, `VendorRiskAssessment`, `assess_vendor_risk` may already be imported — merge, do not duplicate.)

Add the field to `PayableReview`:

```python
    explanations: tuple[Explanation, ...] = ()
```

In `execute`, after the anomaly reconciliation block and before the `approve` trace
record, build the explanations:

```python
        explanations = _explanations(vendor_risk, anomaly)
```

Pass it into `PayableReview(...)`:

```python
            explanations=explanations,
```

Add a module-level helper (next to the class, not a method — it is pure):

```python
def _explanations(
    vendor_risk: VendorRiskAssessment | None, anomaly: AnomalyAssessment | None
) -> tuple[Explanation, ...]:
    built: list[Explanation] = []
    if vendor_risk is not None and vendor_risk.severity is not Severity.NONE:
        built.append(explain_vendor_risk(vendor_risk))
    if anomaly is not None and anomaly.severity is not AnomalySeverity.NONE:
        built.append(explain_anomaly(anomaly))
    return tuple(built)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/application/test_review_payable_anomaly.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/application/review_payable.py tests/unit/application/test_review_payable_anomaly.py
git commit -m "feat(xai): structured explanations on PayableReview"
```

---

### Task 4: `render_explanation` + `apverify-explain` demo CLI

**Files:**
- Modify: `src/apverify/eval/report.py`
- Create: `src/apverify/eval/explanation_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/eval/test_report.py` (append), `tests/unit/eval/test_explanation_cli.py`

**Interfaces:**
- Consumes: `apverify.domain.explanation.Explanation`, `apverify.eval.fusion.{FeatureRow, fit_logistic, explain_fusion}`
- Produces: `render_explanation(explanation: Explanation, console: Console | None = None) -> None`; `apverify-explain` Typer app

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/eval/test_report.py`:

```python
from apverify.domain.explanation import Factor, explanation
from apverify.eval.report import render_explanation


def test_render_explanation_prints_ranked_factors() -> None:
    exp = explanation(
        "fusion",
        "P(correct) 0.30",
        [Factor("arithmetic_passed", "0.00", -1.2, "weight -1.20 × 0.00")],
    )
    console = Console(record=True, width=100)
    render_explanation(exp, console)
    text = console.export_text()
    assert "arithmetic_passed" in text
    assert "P(correct)" in text
```

Create `tests/unit/eval/test_explanation_cli.py`:

```python
from __future__ import annotations

from typer.testing import CliRunner

from apverify.eval.explanation_cli import app


def test_cli_prints_a_fusion_explanation() -> None:
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0
    assert "P(correct)" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_report.py -k render_explanation tests/unit/eval/test_explanation_cli.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement**

Add the import in `src/apverify/eval/report.py`:

```python
from apverify.domain.explanation import Explanation
```

Add the function (near the other renderers):

```python
def render_explanation(explanation: Explanation, console: Console | None = None) -> None:
    console = console or Console()
    console.print(f"[bold]{explanation.source}[/bold] — {explanation.headline}")
    table = Table(title="Ranked factors", title_justify="left")
    table.add_column("Signal", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Contribution", justify="right")
    table.add_column("Detail")
    for factor in explanation.factors:
        table.add_row(
            factor.signal, factor.value, f"{factor.contribution:+.3f}", factor.detail
        )
    console.print(table)
```

Create `src/apverify/eval/explanation_cli.py`:

```python
"""``apverify-explain`` — demo of exact linear attribution on the fusion model.

Fits the interpretable fusion logistic regression on a small deterministic set of
feature rows, then shows the ranked ``weight × feature`` contributions behind one
low-trust field — the auditable "why" behind a fused score.
"""

from __future__ import annotations

import typer
from rich.console import Console

from apverify.eval.fusion import FeatureRow, explain_fusion, fit_logistic
from apverify.eval.report import render_explanation

app = typer.Typer(add_completion=False, help="Explain a fused trust score (linear attribution).")


def _rows() -> list[FeatureRow]:
    # Correctness tracks arithmetic + cross-model; the model learns to weight them.
    rows: list[FeatureRow] = []
    for index in range(12):
        ok = index % 2 == 0
        rows.append(
            FeatureRow(
                label=f"d{index}",
                field="total",
                critic_confidence=0.8,
                verbalized_confidence=0.7,
                cross_check_passed=True,
                arithmetic_passed=ok,
                format_passed=True,
                cross_model_agrees=ok,
                correct=ok,
            )
        )
    return rows


@app.command()
def run() -> None:
    rows = _rows()
    model = fit_logistic(rows)
    suspect = next(row for row in rows if not row.correct)
    render_explanation(explain_fusion(model, suspect), Console())
```

In `pyproject.toml`, add the console script (with the other `apverify-eval-*`):

```toml
apverify-explain = "apverify.eval.explanation_cli:app"
```

Reinstall: `pip install -e .`

- [ ] **Step 4: Run tests + smoke-run**

Run: `pytest tests/unit/eval/test_report.py tests/unit/eval/test_explanation_cli.py -v`
Expected: PASS.
Then: `apverify-explain`
Expected: a ranked factor table headed by `P(correct) …`, with `arithmetic_passed` / `cross_model_agrees` among the dominant contributions.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/report.py src/apverify/eval/explanation_cli.py pyproject.toml tests/unit/eval/test_report.py tests/unit/eval/test_explanation_cli.py
git commit -m "feat(xai): render_explanation + apverify-explain demo"
```

---

## Final verification (after all tasks)

- [ ] `ruff check . && ruff format --check . && mypy --strict src tests` — clean
- [ ] `pytest -q` — all pass; domain layer 100% (`pytest --cov=apverify.domain --cov-report=term-missing`)
- [ ] `apverify-explain` — ranked linear attribution printed
- [ ] README: add an explainability / XAI paragraph + the CLI (small follow-up)

## Spec coverage check

- Honest hybrid (linear fusion + rule conditions) → Tasks 1, 2 ✓
- Unified `Explanation`/`Factor`, ranked by |contribution| → Task 1 ✓
- Exact `weight × feature` fusion attribution → Task 2 ✓
- Pipeline `PayableReview.explanations` for live flags → Task 3 ✓
- `render_explanation` + `apverify-explain` demo → Task 4 ✓
- Acceptance (exact contributions, ranked, pipeline carries explanations, domain 100%) → Tasks 2, 3 + Final verification ✓
```
