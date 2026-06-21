# Cross-detector fraud benchmark (v5 slice 4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One synthesized stream exercising every fraud type, scored by all three shipped detectors together, reporting the combined fraud-catch-rate vs false-positive (the v5 capstone number).

**Architecture:** Pure eval integration — `build_fraud_suite` emits cases each carrying full context (prior-invoice ledger + vendor master + per-vendor history) and a label; `evaluate_fraud_suite` runs the duplicate, BEC, and anomaly detectors on every case, flags a case if any fires, and reports combined metrics plus per-label and per-detector attribution. No new domain/ports/pipeline code, no new dependencies.

**Tech Stack:** Python 3.12, stdlib, frozen dataclasses, typer/rich, pytest.

**Spec:** `docs/superpowers/specs/2026-06-20-v5-cross-detector-benchmark-design.md`

## Global Constraints

- Clean/hexagonal: `eval` may import `domain`; no new domain/application/infrastructure code this slice.
- No new dependencies.
- Determinism: index-derived synthesis, no randomness.
- Detector flag thresholds match the live pipeline: duplicate = any non-DISTINCT match; BEC = severity HIGH; anomaly = severity HIGH or MEDIUM.
- Gates after every task: `ruff check .`, `ruff format --check .`, `mypy --strict src tests`, `pytest`.
- No AI-tells; match surrounding idiom.
- **Git note:** not a git repo, so `git commit` steps are no-ops until `git init`; the per-task checkpoint is the gate suite.

---

### Task 1: Unified suite synthesis — `eval/fraud_suite_synthesis.py`

**Files:**
- Create: `src/apverify/eval/fraud_suite_synthesis.py`
- Test: `tests/unit/eval/test_fraud_suite_synthesis.py`

**Interfaces:**
- Consumes: `apverify.eval.synthetic.{GroundTruth, generate_dataset}`, `apverify.domain.invoice.Invoice`, `apverify.domain.value_objects.Money`, `apverify.domain.fraud.IdentifiedInvoice`, `apverify.domain.vendor_master.KnownVendor`, `apverify.domain.anomaly._round_above`
- Produces:
  - `SuiteCase(invoice: Invoice, priors: tuple[IdentifiedInvoice, ...], master: tuple[KnownVendor, ...], history: tuple[Invoice, ...], label: str, is_fraud: bool)` (frozen)
  - `build_fraud_suite(base: Sequence[GroundTruth]) -> list[SuiteCase]`
  - constants `CLEAN`, `DUP_RESEND`, `DUP_OCR_VARIANT`, `BANK_CHANGE`, `IMPERSONATION`, `AMOUNT_SPIKE`, `THRESHOLD_GAMING`; `LABELS` (all) and `FRAUD_LABELS` (all but CLEAN)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_fraud_suite_synthesis.py`:

```python
from __future__ import annotations

from apverify.eval.fraud_suite_synthesis import (
    CLEAN,
    FRAUD_LABELS,
    LABELS,
    build_fraud_suite,
)
from apverify.eval.synthetic import generate_dataset


def test_every_label_appears_with_full_context() -> None:
    cases = build_fraud_suite(generate_dataset(count=3))
    assert {case.label for case in cases} == set(LABELS)
    for case in cases:
        assert case.priors  # ledger present
        assert case.master  # vendor master present
        assert len(case.history) >= 3  # vendor history present
        assert case.is_fraud == (case.label in FRAUD_LABELS)


def test_clean_case_is_not_a_duplicate_of_the_ledger() -> None:
    # The clean candidate must carry a new invoice number and date so it is not a resend.
    case = next(c for c in build_fraud_suite(generate_dataset(count=1)) if c.label == CLEAN)
    assert all(case.invoice.invoice_number != prior.invoice.invoice_number for prior in case.priors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_fraud_suite_synthesis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.eval.fraud_suite_synthesis'`

- [ ] **Step 3: Implement `eval/fraud_suite_synthesis.py`**

```python
"""Synthesise one labelled stream exercising every fraud type, with full context.

Each case carries the shared prior-invoice ledger, vendor master, and the candidate
vendor's history, so all three detectors can run on it. Frauds are constructed so each
is caught by its own detector while the others stay quiet — the benchmark then measures
whether that isolation actually holds (cross-talk).

Deterministic: fixed transforms of the base, no randomness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from apverify.domain.anomaly import _round_above
from apverify.domain.fraud import IdentifiedInvoice
from apverify.domain.invoice import Invoice
from apverify.domain.value_objects import Money
from apverify.domain.vendor_master import KnownVendor
from apverify.eval.synthetic import GroundTruth

CLEAN = "clean"
DUP_RESEND = "dup_resend"
DUP_OCR_VARIANT = "dup_ocr_variant"
BANK_CHANGE = "bank_change"
IMPERSONATION = "impersonation"
AMOUNT_SPIKE = "amount_spike"
THRESHOLD_GAMING = "threshold_gaming"

LABELS = (
    CLEAN,
    DUP_RESEND,
    DUP_OCR_VARIANT,
    BANK_CHANGE,
    IMPERSONATION,
    AMOUNT_SPIKE,
    THRESHOLD_GAMING,
)
FRAUD_LABELS = tuple(label for label in LABELS if label != CLEAN)

_ATTACKER_BANK = "ACCT-9999"
_NEW_DATE = "04-07-2025"  # a later month, so a same-vendor candidate is not a resend
_OCR_SWAP = str.maketrans({"0": "O", "1": "l"})
_LOOKALIKE = (("e", "3"), ("o", "0"), ("a", "4"), ("i", "1"))
_HISTORY_SPREAD = (Decimal("0.90"), Decimal("0.95"), Decimal("1.00"), Decimal("1.05"), Decimal("1.10"))


@dataclass(frozen=True, slots=True)
class SuiteCase:
    invoice: Invoice
    priors: tuple[IdentifiedInvoice, ...]
    master: tuple[KnownVendor, ...]
    history: tuple[Invoice, ...]
    label: str
    is_fraud: bool


def build_fraud_suite(base: Sequence[GroundTruth]) -> list[SuiteCase]:
    priors = tuple(IdentifiedInvoice(truth.label, truth.invoice) for truth in base)
    names = sorted({truth.invoice.vendor_name for truth in base})
    account_of = {name: f"ACCT-{index:04d}" for index, name in enumerate(names)}
    master = tuple(KnownVendor(name, frozenset({account_of[name]})) for name in names)

    cases: list[SuiteCase] = []
    for truth in base:
        original = truth.invoice
        median = original.total.amount
        known_account = account_of[original.vendor_name]
        history = tuple(_at(original, median * factor) for factor in _HISTORY_SPREAD)

        def case(invoice: Invoice, label: str) -> SuiteCase:
            return SuiteCase(invoice, priors, master, history, label, label != CLEAN)

        fresh = replace(
            original, invoice_number=f"{original.invoice_number}-N", invoice_date=_NEW_DATE
        )
        cases.extend(
            [
                case(replace(fresh, bank_account=known_account, total=_money(median * Decimal("1.02")), subtotal=_money(median * Decimal("1.02"))), CLEAN),
                case(original, DUP_RESEND),
                case(replace(original, invoice_number=original.invoice_number.translate(_OCR_SWAP)), DUP_OCR_VARIANT),
                case(replace(fresh, bank_account=_ATTACKER_BANK), BANK_CHANGE),
                case(replace(fresh, vendor_name=_typosquat(original.vendor_name), bank_account=_ATTACKER_BANK), IMPERSONATION),
                case(_at(fresh, median * 10), AMOUNT_SPIKE),
                case(_at(fresh, _just_under_round(median)), THRESHOLD_GAMING),
            ]
        )
    return cases


def _at(invoice: Invoice, total: Decimal) -> Invoice:
    money = _money(total)
    return replace(invoice, total=money, subtotal=money)


def _money(total: Decimal) -> Money:
    return Money(total.quantize(Decimal("0.01")))


def _just_under_round(median: Decimal) -> Decimal:
    limit = Decimal(str(_round_above(float(median))))
    return (limit * Decimal("0.995")).quantize(Decimal("1"))


def _typosquat(name: str) -> str:
    lowered = name.lower()
    for original, lookalike in _LOOKALIKE:
        index = lowered.find(original)
        if index != -1:
            return name[:index] + lookalike + name[index + 1 :]
    return name + "X"
```

Note: the inner `case` closure captures `priors`, `master`, `history` per base invoice — defining it inside the loop is intentional so each case binds that vendor's history. If your linter flags the loop-closure (`B023`), hoist `case` to a module-level helper taking the context explicitly; behaviour must be identical.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/eval/test_fraud_suite_synthesis.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/fraud_suite_synthesis.py tests/unit/eval/test_fraud_suite_synthesis.py
git commit -m "feat(eval): unified cross-detector fraud suite synthesis"
```

---

### Task 2: Combined evaluator — `eval/fraud_suite_eval.py`

**Files:**
- Create: `src/apverify/eval/fraud_suite_eval.py`
- Test: `tests/unit/eval/test_fraud_suite_eval.py`

**Interfaces:**
- Consumes: `apverify.domain.fraud.find_duplicates`, `apverify.domain.vendor_master.{assess_vendor_risk, Severity}`, `apverify.domain.anomaly.{RobustAnomalyDetector, AnomalySeverity}`, `apverify.eval.fraud_suite_synthesis.SuiteCase`
- Produces:
  - `FraudSuiteReport(case_count: int, fraud_count: int, catch_rate: float, false_positive_rate: float, precision: float, per_label: dict[str, float], per_detector: dict[str, int])` (frozen)
  - `evaluate_fraud_suite(cases: Sequence[SuiteCase]) -> FraudSuiteReport`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_fraud_suite_eval.py`:

```python
from __future__ import annotations

from apverify.eval.fraud_suite_eval import evaluate_fraud_suite
from apverify.eval.fraud_suite_synthesis import build_fraud_suite
from apverify.eval.synthetic import generate_dataset


def test_every_fraud_is_caught_with_no_false_positives() -> None:
    report = evaluate_fraud_suite(build_fraud_suite(generate_dataset(count=5)))
    assert report.catch_rate == 1.0
    assert report.false_positive_rate == 0.0


def test_each_fraud_type_is_attributed_to_its_own_detector() -> None:
    report = evaluate_fraud_suite(build_fraud_suite(generate_dataset(count=5)))
    assert report.per_label["dup_resend"] == 1.0
    assert report.per_label["bank_change"] == 1.0
    assert report.per_label["amount_spike"] == 1.0
    assert report.per_label["clean"] == 0.0
    # Each detector caught at least its own frauds.
    assert report.per_detector["duplicate"] >= 1
    assert report.per_detector["bec"] >= 1
    assert report.per_detector["anomaly"] >= 1


def test_empty_cases_yield_a_zeroed_report() -> None:
    report = evaluate_fraud_suite([])
    assert report.case_count == 0
    assert report.catch_rate == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_fraud_suite_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.eval.fraud_suite_eval'`

- [ ] **Step 3: Implement `eval/fraud_suite_eval.py`**

```python
"""Score the three fraud detectors together over the unified suite.

Every case is run through all three detectors at the live-pipeline thresholds; a case is
flagged if any fires. The combined catch-rate and the system-wide false-positive rate
(a clean invoice flagged by *any* detector — cross-talk) are the capstone numbers, with
a per-label catch table and per-detector attribution beneath them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.domain.anomaly import AnomalySeverity, RobustAnomalyDetector
from apverify.domain.fraud import find_duplicates
from apverify.domain.vendor_master import Severity, assess_vendor_risk
from apverify.eval.fraud_suite_synthesis import SuiteCase

_DETECTORS = ("duplicate", "bec", "anomaly")


@dataclass(frozen=True, slots=True)
class FraudSuiteReport:
    case_count: int
    fraud_count: int
    catch_rate: float
    false_positive_rate: float
    precision: float
    per_label: dict[str, float]
    per_detector: dict[str, int]


def evaluate_fraud_suite(cases: Sequence[SuiteCase]) -> FraudSuiteReport:
    anomaly = RobustAnomalyDetector()
    scored = [(case, _fired(case, anomaly)) for case in cases]

    frauds = [(case, fired) for case, fired in scored if case.is_fraud]
    cleans = [(case, fired) for case, fired in scored if not case.is_fraud]
    flagged = [(case, fired) for case, fired in scored if fired]

    return FraudSuiteReport(
        case_count=len(cases),
        fraud_count=len(frauds),
        catch_rate=_rate([1 for _, fired in frauds if fired], frauds),
        false_positive_rate=_rate([1 for _, fired in cleans if fired], cleans),
        precision=_rate([1 for case, _ in flagged if case.is_fraud], flagged),
        per_label=_per_label(scored),
        per_detector=_per_detector(scored),
    )


def _fired(case: SuiteCase, anomaly: RobustAnomalyDetector) -> frozenset[str]:
    fired: set[str] = set()
    if find_duplicates(case.invoice, case.priors):
        fired.add("duplicate")
    if assess_vendor_risk(case.invoice, case.master).severity is Severity.HIGH:
        fired.add("bec")
    if anomaly.score(case.invoice, case.history).severity in (
        AnomalySeverity.HIGH,
        AnomalySeverity.MEDIUM,
    ):
        fired.add("anomaly")
    return frozenset(fired)


def _rate(hits: Sequence[object], population: Sequence[object]) -> float:
    return len(hits) / len(population) if population else 0.0


def _per_label(scored: Sequence[tuple[SuiteCase, frozenset[str]]]) -> dict[str, float]:
    labels = sorted({case.label for case, _ in scored})
    return {
        label: _rate(
            [1 for case, fired in scored if case.label == label and fired],
            [1 for case, _ in scored if case.label == label],
        )
        for label in labels
    }


def _per_detector(scored: Sequence[tuple[SuiteCase, frozenset[str]]]) -> dict[str, int]:
    return {
        detector: sum(1 for case, fired in scored if case.is_fraud and detector in fired)
        for detector in _DETECTORS
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/eval/test_fraud_suite_eval.py -v`
Expected: PASS (all three).

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/fraud_suite_eval.py tests/unit/eval/test_fraud_suite_eval.py
git commit -m "feat(eval): combined cross-detector evaluation"
```

---

### Task 3: `render_fraud_suite` + `apverify-eval-fraud-suite` CLI

**Files:**
- Modify: `src/apverify/eval/report.py`
- Create: `src/apverify/eval/fraud_suite_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/eval/test_report.py` (append), `tests/unit/eval/test_fraud_suite_cli.py`

**Interfaces:**
- Consumes: `apverify.eval.fraud_suite_eval.FraudSuiteReport`, `evaluate_fraud_suite`, `build_fraud_suite`, `generate_dataset`
- Produces: `render_fraud_suite(report: FraudSuiteReport, console: Console | None = None) -> None`; `apverify-eval-fraud-suite` Typer app with `--count`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/eval/test_report.py`:

```python
from apverify.eval.fraud_suite_eval import FraudSuiteReport
from apverify.eval.report import render_fraud_suite


def test_render_fraud_suite_prints_combined_and_per_label() -> None:
    report = FraudSuiteReport(
        case_count=35,
        fraud_count=30,
        catch_rate=1.0,
        false_positive_rate=0.0,
        precision=1.0,
        per_label={"dup_resend": 1.0, "clean": 0.0},
        per_detector={"duplicate": 10, "bec": 10, "anomaly": 10},
    )
    console = Console(record=True, width=100)
    render_fraud_suite(report, console)
    text = console.export_text()
    assert "dup_resend" in text
    assert "fraud" in text.lower()
```

Create `tests/unit/eval/test_fraud_suite_cli.py`:

```python
from __future__ import annotations

from typer.testing import CliRunner

from apverify.eval.fraud_suite_cli import app


def test_cli_runs_the_fraud_suite() -> None:
    result = CliRunner().invoke(app, ["--count", "5"])
    assert result.exit_code == 0
    assert "fraud" in result.stdout.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_report.py -k render_fraud_suite tests/unit/eval/test_fraud_suite_cli.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement**

Add the import in `src/apverify/eval/report.py` (with the eval imports):

```python
from apverify.eval.fraud_suite_eval import FraudSuiteReport
```

Add the function (near `render_fraud`):

```python
def render_fraud_suite(report: FraudSuiteReport, console: Console | None = None) -> None:
    console = console or Console()
    if report.case_count == 0:
        console.print("[yellow]No fraud-suite cases to evaluate.[/yellow]")
        return

    console.print(
        f"[bold]Cross-detector fraud suite[/bold] (n={report.case_count}, "
        f"{report.fraud_count} fraudulent): the combined layer catches "
        f"[green]{report.catch_rate:.0%}[/green] of fraud at "
        f"[green]{report.false_positive_rate:.0%}[/green] false-positive "
        f"(precision {report.precision:.0%})."
    )

    by_label = Table(title="Catch rate by fraud type", title_justify="left")
    by_label.add_column("Label", style="cyan")
    by_label.add_column("Flagged", justify="right")
    for label, rate in report.per_label.items():
        by_label.add_row(label, f"{rate:.0%}")
    console.print(by_label)

    by_detector = Table(title="Frauds caught per detector", title_justify="left")
    by_detector.add_column("Detector", style="cyan")
    by_detector.add_column("Caught", justify="right")
    for detector, count in report.per_detector.items():
        by_detector.add_row(detector, str(count))
    console.print(by_detector)
```

Create `src/apverify/eval/fraud_suite_cli.py`:

```python
"""``apverify-eval-fraud-suite`` — the combined cross-detector fraud benchmark.

Runs the duplicate, BEC, and anomaly detectors together over one synthesized stream and
reports the combined catch-rate vs false-positive. Synthetic only.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.fraud_suite_eval import evaluate_fraud_suite
from apverify.eval.fraud_suite_synthesis import build_fraud_suite
from apverify.eval.report import render_fraud_suite
from apverify.eval.synthetic import generate_dataset

app = typer.Typer(add_completion=False, help="Combined cross-detector fraud benchmark.")


@app.command()
def run(count: Annotated[int, typer.Option(help="Synthetic base invoices.")] = 25) -> None:
    report = evaluate_fraud_suite(build_fraud_suite(generate_dataset(count=count)))
    render_fraud_suite(report, Console())
```

In `pyproject.toml`, add the console script (with the other `apverify-eval-*`):

```toml
apverify-eval-fraud-suite = "apverify.eval.fraud_suite_cli:app"
```

Reinstall: `pip install -e .`

- [ ] **Step 4: Run tests + smoke-run**

Run: `pytest tests/unit/eval/test_report.py tests/unit/eval/test_fraud_suite_cli.py -v`
Expected: PASS.
Then: `apverify-eval-fraud-suite --count 25`
Expected: combined catch-rate 100% @ 0% false-positive; per-label table with each fraud at 100%, clean at 0%; per-detector counts non-zero for all three.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/report.py src/apverify/eval/fraud_suite_cli.py pyproject.toml tests/unit/eval/test_report.py tests/unit/eval/test_fraud_suite_cli.py
git commit -m "feat(cli): apverify-eval-fraud-suite combined benchmark"
```

---

## Final verification (after all tasks)

- [ ] `ruff check . && ruff format --check . && mypy --strict src tests` — clean
- [ ] `pytest -q` — all pass
- [ ] `apverify-eval-fraud-suite --count 25` — combined catch 100% @ 0% FP; every fraud type 100%, clean 0%; each detector attributed to its own fraud type
- [ ] README: add a cross-detector / combined-fraud-layer paragraph + the CLI (small follow-up)

## Spec coverage check

- Eval-only integration of the three detectors → Tasks 1–3 ✓
- Unified dataset, full context per case → Task 1 ✓
- All three detectors per case, union flag + attribution → Task 2 ✓
- Combined catch / system-wide FP / precision / per-label / per-detector → Task 2 ✓
- Detector thresholds match the live pipeline → Task 2 (`_fired`) ✓
- Cross-talk measured (clean flagged by any detector) → Task 2 (`false_positive_rate`) ✓
- CLI + render → Task 3 ✓
- Acceptance (100% catch, 0% FP, per-detector attribution) → Tasks 2 + Final verification ✓
```
