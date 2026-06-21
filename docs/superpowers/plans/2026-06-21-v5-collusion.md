# Collusion detection (v5 slice 6) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect collusion from approval *behaviour* — an approver funneling one vendor, approvals just under the approver's limit, rubber-stamped within seconds — over an approval log, with a synthetic catch-rate vs false-positive benchmark.

**Architecture:** A pure `domain/collusion.py` detector over a sequence of approval records (cross-record, batch — not per-invoice), plus an eval benchmark. No new dependencies.

**Tech Stack:** Python 3.12, stdlib (`datetime`, `statistics`), frozen dataclasses, typer/rich, pytest.

**Spec:** `docs/superpowers/specs/2026-06-21-v5-collusion-design.md`

## Global Constraints

- Clean/hexagonal: `domain` imports only stdlib + `domain`; `eval` imports inward.
- No new dependencies.
- Determinism: synthetic datetimes derived from index (no wall-clock); the domain takes datetimes as data.
- `detect_collusion` returns every pair meeting the min-approvals floor, ranked by score; `severity` is NONE below `flag_score` (a pair is "flagged" when `severity != NONE`).
- Gates after every task: `ruff check .`, `ruff format --check .`, `mypy --strict src tests`, `pytest`. **Domain layer 100% coverage.**
- No AI-tells; match surrounding idiom.
- **Git note:** not a git repo, so `git commit` steps are no-ops; the per-task checkpoint is the gate suite.

---

### Task 1: Domain collusion detector

**Files:**
- Create: `src/apverify/domain/collusion.py`
- Test: `tests/unit/domain/test_collusion.py`

**Interfaces:**
- Consumes: `apverify.domain.value_objects.Money`
- Produces:
  - `class CollusionSeverity(Enum)`: `NONE`, `MEDIUM`, `HIGH`
  - `ApprovalRecord(submitter: str, approver: str, vendor: str, amount: Money, submitted_at: datetime, approved_at: datetime, approver_limit: Money)` (frozen)
  - `CollusionSignal(approver: str, vendor: str, concentration: float, under_limit_rate: float, rubber_stamp_rate: float, score: float, severity: CollusionSeverity, reason: str)` (frozen)
  - `detect_collusion(records: Sequence[ApprovalRecord], *, under_limit_band: float = 0.05, rubber_stamp_seconds: float = 60.0, flag_score: float = 0.6, min_approvals: int = 3) -> list[CollusionSignal]`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/domain/test_collusion.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from apverify.domain.collusion import (
    ApprovalRecord,
    CollusionSeverity,
    detect_collusion,
)
from apverify.domain.value_objects import Money

_BASE = datetime(2025, 6, 1, 9, 0, 0)
_LIMIT = Money(Decimal("50000"))


def _record(approver: str, vendor: str, amount: str, *, day: int, latency_s: float) -> ApprovalRecord:
    submitted = _BASE + timedelta(days=day)
    return ApprovalRecord(
        submitter="bob",
        approver=approver,
        vendor=vendor,
        amount=Money(Decimal(amount)),
        submitted_at=submitted,
        approved_at=submitted + timedelta(seconds=latency_s),
        approver_limit=_LIMIT,
    )


def _colluding(approver: str, vendor: str, count: int) -> list[ApprovalRecord]:
    return [_record(approver, vendor, "49500", day=j, latency_s=10) for j in range(count)]


def test_a_funneled_just_under_rubber_stamped_pair_is_high() -> None:
    signals = detect_collusion(_colluding("alice", "acme", 5))
    assert signals[0].severity is CollusionSeverity.HIGH
    assert signals[0].approver == "alice"
    assert signals[0].concentration == 1.0


def test_a_diverse_normal_approver_is_not_flagged() -> None:
    records = [
        _record("carol", f"vendor{j % 2}", str(10000 + j * 2000), day=j, latency_s=18000)
        for j in range(6)
    ]
    signals = detect_collusion(records)
    assert all(s.severity is CollusionSeverity.NONE for s in signals)


def test_a_pair_below_the_minimum_is_not_scored() -> None:
    signals = detect_collusion(_colluding("alice", "acme", 2))  # only 2 < min 3
    assert signals == []


def test_reason_names_the_patterns() -> None:
    signals = detect_collusion(_colluding("alice", "acme", 5))
    assert "alice" in signals[0].reason and "acme" in signals[0].reason
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/domain/test_collusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.domain.collusion'`

- [ ] **Step 3: Implement `domain/collusion.py`**

```python
"""Collusion detection from approval behaviour.

Collusion shows up not in invoice text but in *who approves what*: an approver who
funnels one vendor, clears amounts parked just under their own authorization limit, and
rubber-stamps them within seconds. This detects those patterns over a log of approval
records — cross-record and batch, not per-invoice. Pure domain logic; timestamps are
passed in as data, so there is no wall-clock here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from apverify.domain.value_objects import Money


class CollusionSeverity(Enum):
    NONE = "none"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    submitter: str
    approver: str
    vendor: str
    amount: Money
    submitted_at: datetime
    approved_at: datetime
    approver_limit: Money


@dataclass(frozen=True, slots=True)
class CollusionSignal:
    approver: str
    vendor: str
    concentration: float
    under_limit_rate: float
    rubber_stamp_rate: float
    score: float
    severity: CollusionSeverity
    reason: str


def detect_collusion(
    records: Sequence[ApprovalRecord],
    *,
    under_limit_band: float = 0.05,
    rubber_stamp_seconds: float = 60.0,
    flag_score: float = 0.6,
    min_approvals: int = 3,
) -> list[CollusionSignal]:
    approver_totals = Counter(record.approver for record in records)
    pairs: dict[tuple[str, str], list[ApprovalRecord]] = {}
    for record in records:
        pairs.setdefault((record.approver, record.vendor), []).append(record)

    signals: list[CollusionSignal] = []
    for (approver, vendor), group in pairs.items():
        if len(group) < min_approvals:
            continue
        concentration = len(group) / approver_totals[approver]
        under_limit_rate = _rate(group, lambda r: _just_under(r.amount, r.approver_limit, under_limit_band))
        rubber_stamp_rate = _rate(
            group, lambda r: (r.approved_at - r.submitted_at).total_seconds() <= rubber_stamp_seconds
        )
        score = (concentration + under_limit_rate + rubber_stamp_rate) / 3
        severity = (
            CollusionSeverity.HIGH
            if score >= 0.8
            else CollusionSeverity.MEDIUM
            if score >= flag_score
            else CollusionSeverity.NONE
        )
        signals.append(
            CollusionSignal(
                approver=approver,
                vendor=vendor,
                concentration=round(concentration, 4),
                under_limit_rate=round(under_limit_rate, 4),
                rubber_stamp_rate=round(rubber_stamp_rate, 4),
                score=round(score, 4),
                severity=severity,
                reason=(
                    f"approver {approver} cleared {concentration:.0%} of vendor {vendor}'s "
                    f"invoices, {under_limit_rate:.0%} just under limit, "
                    f"{rubber_stamp_rate:.0%} rubber-stamped"
                ),
            )
        )
    return sorted(signals, key=lambda signal: signal.score, reverse=True)


def _rate(group: Sequence[ApprovalRecord], predicate: object) -> float:
    matches = sum(1 for record in group if predicate(record))  # type: ignore[operator]
    return matches / len(group)


def _just_under(amount: Money, limit: Money, band: float) -> bool:
    if limit.amount <= 0:
        return False
    gap = (limit.amount - amount.amount) / limit.amount
    return 0 <= gap <= band
```

Note: `_rate` takes a callable; if mypy's `object` annotation is awkward, type it as `Callable[[ApprovalRecord], bool]` from `collections.abc` and drop the `type: ignore`. Prefer the typed version.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/domain/test_collusion.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/domain/collusion.py tests/unit/domain/test_collusion.py
git commit -m "feat(collusion): behavioral approver-vendor detector"
```

---

### Task 2: Collusion synthesis — `eval/collusion_synthesis.py`

**Files:**
- Create: `src/apverify/eval/collusion_synthesis.py`
- Test: `tests/unit/eval/test_collusion_synthesis.py`

**Interfaces:**
- Consumes: `apverify.domain.collusion.ApprovalRecord`, `apverify.domain.value_objects.Money`
- Produces: `build_collusion_log(pairs: int = 6, per_pair: int = 8) -> tuple[list[ApprovalRecord], dict[tuple[str, str], bool]]`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_collusion_synthesis.py`:

```python
from __future__ import annotations

from apverify.eval.collusion_synthesis import build_collusion_log


def test_log_has_both_colluding_and_normal_pairs() -> None:
    records, truth = build_collusion_log(pairs=6, per_pair=8)
    assert records
    assert any(truth.values())  # at least one colluding pair
    assert not all(truth.values())  # and at least one normal pair


def test_every_truth_pair_has_enough_records_to_score() -> None:
    records, truth = build_collusion_log(pairs=4, per_pair=8)
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        counts[(record.approver, record.vendor)] = counts.get((record.approver, record.vendor), 0) + 1
    for pair in truth:
        assert counts[pair] >= 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_collusion_synthesis.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `eval/collusion_synthesis.py`**

```python
"""Synthesise a labelled approval log for the collusion benchmark.

Colluding pairs: one approver funnels one vendor — every approval just under the
approver's limit, approved within seconds. Normal pairs: an approver spreads work across
vendors, with varied amounts well below the limit and realistic (hours) approval latency.

Deterministic: timestamps and amounts derive from the index, no randomness.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from apverify.domain.collusion import ApprovalRecord
from apverify.domain.value_objects import Money

_BASE = datetime(2025, 6, 1, 9, 0, 0)
_LIMIT = Money(Decimal("50000"))


def build_collusion_log(
    pairs: int = 6, per_pair: int = 8
) -> tuple[list[ApprovalRecord], dict[tuple[str, str], bool]]:
    records: list[ApprovalRecord] = []
    truth: dict[tuple[str, str], bool] = {}
    for index in range(pairs):
        approver = f"approver{index}"
        if index % 2 == 0:
            vendor = f"vendor{index}"
            records.extend(_colluding(approver, vendor, per_pair))
            truth[(approver, vendor)] = True
        else:
            for variant in range(2):
                vendor = f"vendor{index}_{variant}"
                records.extend(_normal(approver, vendor, per_pair))
                truth[(approver, vendor)] = False
    return records, truth


def _colluding(approver: str, vendor: str, count: int) -> list[ApprovalRecord]:
    return [
        ApprovalRecord(
            submitter="submitter",
            approver=approver,
            vendor=vendor,
            amount=Money(Decimal("49500")),  # just under the 50000 limit
            submitted_at=_BASE + timedelta(days=day),
            approved_at=_BASE + timedelta(days=day, seconds=10),  # rubber-stamped
            approver_limit=_LIMIT,
        )
        for day in range(count)
    ]


def _normal(approver: str, vendor: str, count: int) -> list[ApprovalRecord]:
    return [
        ApprovalRecord(
            submitter="submitter",
            approver=approver,
            vendor=vendor,
            amount=Money(Decimal(str(10000 + day * 2000))),  # varied, well below limit
            submitted_at=_BASE + timedelta(days=day),
            approved_at=_BASE + timedelta(days=day, hours=5),  # real review latency
            approver_limit=_LIMIT,
        )
        for day in range(count)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/eval/test_collusion_synthesis.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/collusion_synthesis.py tests/unit/eval/test_collusion_synthesis.py
git commit -m "feat(eval): synthetic collusion approval log"
```

---

### Task 3: Collusion metrics — `eval/collusion_eval.py`

**Files:**
- Create: `src/apverify/eval/collusion_eval.py`
- Test: `tests/unit/eval/test_collusion_eval.py`

**Interfaces:**
- Consumes: `apverify.domain.collusion.{detect_collusion, CollusionSeverity, ApprovalRecord}`, `apverify.eval.fusion.auroc`
- Produces:
  - `CollusionReport(pair_count: int, colluding_count: int, catch_rate: float, false_positive_rate: float, precision: float, auroc: float)` (frozen)
  - `evaluate_collusion(records: Sequence[ApprovalRecord], truth: dict[tuple[str, str], bool]) -> CollusionReport`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_collusion_eval.py`:

```python
from __future__ import annotations

from apverify.eval.collusion_eval import evaluate_collusion
from apverify.eval.collusion_synthesis import build_collusion_log


def test_colluding_pairs_are_caught_with_no_false_positives() -> None:
    records, truth = build_collusion_log(pairs=6, per_pair=8)
    report = evaluate_collusion(records, truth)
    assert report.catch_rate == 1.0
    assert report.false_positive_rate == 0.0
    assert report.auroc >= 0.9


def test_empty_log_yields_a_zeroed_report() -> None:
    report = evaluate_collusion([], {})
    assert report.pair_count == 0
    assert report.catch_rate == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_collusion_eval.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `eval/collusion_eval.py`**

```python
"""Score the collusion detector against a labelled approval log."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.domain.collusion import ApprovalRecord, CollusionSeverity, detect_collusion
from apverify.eval.fusion import auroc


@dataclass(frozen=True, slots=True)
class CollusionReport:
    pair_count: int
    colluding_count: int
    catch_rate: float
    false_positive_rate: float
    precision: float
    auroc: float


def evaluate_collusion(
    records: Sequence[ApprovalRecord], truth: dict[tuple[str, str], bool]
) -> CollusionReport:
    signals = detect_collusion(records)
    score_by_pair = {(signal.approver, signal.vendor): signal.score for signal in signals}
    flagged = {
        (signal.approver, signal.vendor)
        for signal in signals
        if signal.severity is not CollusionSeverity.NONE
    }

    colluding = [pair for pair, is_collusion in truth.items() if is_collusion]
    normal = [pair for pair, is_collusion in truth.items() if not is_collusion]
    flagged_truth = [pair for pair in truth if pair in flagged]
    samples = [(score_by_pair.get(pair, 0.0), is_collusion) for pair, is_collusion in truth.items()]

    return CollusionReport(
        pair_count=len(truth),
        colluding_count=len(colluding),
        catch_rate=_rate([p for p in colluding if p in flagged], colluding),
        false_positive_rate=_rate([p for p in normal if p in flagged], normal),
        precision=_rate([p for p in flagged_truth if truth[p]], flagged_truth),
        auroc=auroc(samples),
    )


def _rate(hits: Sequence[object], population: Sequence[object]) -> float:
    return len(hits) / len(population) if population else 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/eval/test_collusion_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/collusion_eval.py tests/unit/eval/test_collusion_eval.py
git commit -m "feat(eval): collusion benchmark metrics"
```

---

### Task 4: `render_collusion` + `apverify-eval-collusion` CLI

**Files:**
- Modify: `src/apverify/eval/report.py`
- Create: `src/apverify/eval/collusion_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/eval/test_report.py` (append), `tests/unit/eval/test_collusion_cli.py`

**Interfaces:**
- Consumes: `apverify.eval.collusion_eval.CollusionReport`, `evaluate_collusion`, `build_collusion_log`
- Produces: `render_collusion(report: CollusionReport, console: Console | None = None) -> None`; `apverify-eval-collusion` Typer app

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/eval/test_report.py`:

```python
from apverify.eval.collusion_eval import CollusionReport
from apverify.eval.report import render_collusion


def test_render_collusion_prints_headline() -> None:
    report = CollusionReport(
        pair_count=9,
        colluding_count=3,
        catch_rate=1.0,
        false_positive_rate=0.0,
        precision=1.0,
        auroc=1.0,
    )
    console = Console(record=True, width=100)
    render_collusion(report, console)
    text = console.export_text()
    assert "collusion" in text.lower()
```

Create `tests/unit/eval/test_collusion_cli.py`:

```python
from __future__ import annotations

from typer.testing import CliRunner

from apverify.eval.collusion_cli import app


def test_cli_runs_the_collusion_benchmark() -> None:
    result = CliRunner().invoke(app, ["--pairs", "6"])
    assert result.exit_code == 0
    assert "collusion" in result.stdout.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_report.py -k render_collusion tests/unit/eval/test_collusion_cli.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement**

Add the import in `src/apverify/eval/report.py`:

```python
from apverify.eval.collusion_eval import CollusionReport
```

Add the function (near the other renderers):

```python
def render_collusion(report: CollusionReport, console: Console | None = None) -> None:
    console = console or Console()
    if report.pair_count == 0:
        console.print("[yellow]No approver-vendor pairs to evaluate.[/yellow]")
        return
    console.print(
        f"[bold]Collusion detection[/bold] ({report.pair_count} pairs, "
        f"{report.colluding_count} colluding): caught "
        f"[green]{report.catch_rate:.0%}[/green] at "
        f"[green]{report.false_positive_rate:.0%}[/green] false-positive "
        f"(precision {report.precision:.0%}, AUROC {report.auroc:.3f})."
    )
```

Create `src/apverify/eval/collusion_cli.py`:

```python
"""``apverify-eval-collusion`` — behavioral collusion benchmark over an approval log."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.collusion_eval import evaluate_collusion
from apverify.eval.collusion_synthesis import build_collusion_log
from apverify.eval.report import render_collusion

app = typer.Typer(add_completion=False, help="Behavioral collusion-detection benchmark.")


@app.command()
def run(
    pairs: Annotated[int, typer.Option(help="Approver-vendor pairs.")] = 6,
    per_pair: Annotated[int, typer.Option(help="Approvals per pair.")] = 8,
) -> None:
    records, truth = build_collusion_log(pairs=pairs, per_pair=per_pair)
    render_collusion(evaluate_collusion(records, truth), Console())
```

In `pyproject.toml`, add the console script:

```toml
apverify-eval-collusion = "apverify.eval.collusion_cli:app"
```

Reinstall: `pip install -e .`

- [ ] **Step 4: Run tests + smoke-run**

Run: `pytest tests/unit/eval/test_report.py tests/unit/eval/test_collusion_cli.py -v`
Expected: PASS.
Then: `apverify-eval-collusion --pairs 6`
Expected: caught 100% at 0% false-positive, AUROC 1.000.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/report.py src/apverify/eval/collusion_cli.py pyproject.toml tests/unit/eval/test_report.py tests/unit/eval/test_collusion_cli.py
git commit -m "feat(cli): apverify-eval-collusion benchmark"
```

---

## Final verification (after all tasks)

- [ ] `ruff check . && ruff format --check . && mypy --strict src tests` — clean
- [ ] `pytest -q` — all pass; domain layer 100% (`pytest --cov=apverify.domain --cov-report=term-missing`)
- [ ] `apverify-eval-collusion --pairs 6` — caught 100% @ 0% false-positive, AUROC 1.000
- [ ] README: add a collusion paragraph + the CLI (small follow-up). **This completes v5's six sub-projects.**

## Spec coverage check

- Behavioral detector (concentration + under-limit + rubber-stamp) → Task 1 ✓
- ApprovalRecord data model → Task 1 ✓
- min-approvals abstention → Task 1 ✓
- synthetic colluding + normal log → Task 2 ✓
- catch / FP / precision / AUROC over pairs → Task 3 ✓
- render + CLI → Task 4 ✓
- Acceptance (≥90% catch @ 0% FP, AUROC ≥0.9, single-approval never flagged, domain 100%) → Tasks 1, 3 + Final verification ✓
```
