# Anomaly detection (v5 slice 3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flag invoices statistically unusual for their vendor (amount spike, threshold gaming) with an explainable pure-statistics detector wired into the pipeline, plus a benchmark pitting it head-to-head against an optional Isolation Forest.

**Architecture:** A pure `domain/anomaly.py` detector (median/MAD robust-z + threshold-proximity) is the production default and the pipeline's anomaly stage. scikit-learn's Isolation Forest is an eval-only optional challenger, imported guarded so its absence degrades the benchmark to pure-only and it never touches the production path.

**Tech Stack:** Python 3.12, `statistics`/`math` (stdlib), frozen dataclasses, optional `scikit-learn` extra, pytest. No new *required* dependency.

**Spec:** `docs/superpowers/specs/2026-06-20-v5-anomaly-detection-design.md`

## Global Constraints

- Clean/hexagonal: `domain` imports only stdlib + `domain`; `application` imports `domain`; outer layers import inward. (dependency rule)
- No new *required* dependency. scikit-learn is an **optional extra**, used **only** by the benchmark, imported guarded. It never appears on the production path.
- Determinism: no wall-clock; synthetic values derive from index. (Isolation Forest uses a fixed `random_state=0`.)
- Detector is pure and total: insufficient history (default < 3 priors) → abstain; MAD floored so robust-z never divides by zero.
- Anomaly score ∈ [0,1]; `score = max(1 - exp(-robust_z / sensitivity), threshold_proximity)`; severity ≥0.8 HIGH, ≥0.5 MEDIUM, else NONE.
- Pipeline: HIGH → HOLD, MEDIUM → HUMAN_REVIEW, NONE → unchanged; never lowers a decision.
- Gates after every task: `ruff check .`, `ruff format --check .`, `mypy --strict src tests`, `pytest`. **Domain layer 100% coverage.**
- No AI-tells; match surrounding idiom.
- **Git note:** not a git repo, so `git commit` steps are no-ops until `git init`; the per-task checkpoint is the gate suite.

---

### Task 1: Pure anomaly detector + features

**Files:**
- Create: `src/apverify/domain/anomaly.py`
- Test: `tests/unit/domain/test_anomaly.py`

**Interfaces:**
- Consumes: `apverify.domain.invoice.Invoice`
- Produces:
  - `class AnomalySeverity(Enum)`: `NONE`, `MEDIUM`, `HIGH`
  - `AnomalyFeatures(amount_robust_z: float, threshold_proximity: float, history_size: int)` (frozen)
  - `AnomalyAssessment(score: float, severity: AnomalySeverity, top_feature: str, reason: str)` (frozen)
  - `extract_features(invoice: Invoice, history: Sequence[Invoice], band: float = 0.05) -> AnomalyFeatures`
  - `RobustAnomalyDetector(min_history: int = 3, sensitivity: float = 3.0, band: float = 0.05, high: float = 0.8, medium: float = 0.5)` with `score(invoice: Invoice, history: Sequence[Invoice]) -> AnomalyAssessment`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/domain/test_anomaly.py`:

```python
from __future__ import annotations

from decimal import Decimal

from apverify.domain.anomaly import (
    AnomalySeverity,
    RobustAnomalyDetector,
    extract_features,
)
from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.value_objects import Money


def _invoice(total: str) -> Invoice:
    amount = Money(Decimal(total))
    return Invoice(
        vendor_name="ACME Steel Pvt Ltd",
        invoice_number="INV-1",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=amount,
        tax=Money(Decimal("0")),
        total=amount,
        line_items=(LineItem("Widget", 1, amount, amount),),
        tax_breakdown=TaxBreakdown(),
        vendor_gstin="",
        purchase_order_ref="",
    )


def _history(totals: list[str]) -> list[Invoice]:
    return [_invoice(total) for total in totals]


_NORMAL_HISTORY = _history(["90", "95", "100", "105", "110"])
_DETECTOR = RobustAnomalyDetector()


def test_in_range_amount_is_not_anomalous() -> None:
    result = _DETECTOR.score(_invoice("102"), _NORMAL_HISTORY)
    assert result.severity is AnomalySeverity.NONE


def test_amount_spike_is_high_severity() -> None:
    result = _DETECTOR.score(_invoice("1000"), _NORMAL_HISTORY)
    assert result.severity is AnomalySeverity.HIGH
    assert result.top_feature == "amount_spike"
    assert "median" in result.reason


def test_amount_just_under_a_round_limit_is_flagged_for_gaming() -> None:
    history = _history(["9000", "9100", "9200", "9300", "9150"])
    result = _DETECTOR.score(_invoice("9950"), history)
    assert result.severity is AnomalySeverity.HIGH
    assert result.top_feature == "threshold_gaming"
    assert "approval limit" in result.reason


def test_insufficient_history_abstains() -> None:
    result = _DETECTOR.score(_invoice("1000"), _history(["100", "100"]))
    assert result.severity is AnomalySeverity.NONE
    assert result.score == 0.0
    assert "insufficient history" in result.reason


def test_identical_history_does_not_divide_by_zero() -> None:
    result = _DETECTOR.score(_invoice("100"), _history(["100", "100", "100", "100"]))
    assert result.severity is AnomalySeverity.NONE  # same as history, no anomaly


def test_extract_features_with_no_history_reports_zero_z() -> None:
    features = extract_features(_invoice("100"), [])
    assert features.amount_robust_z == 0.0
    assert features.history_size == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/domain/test_anomaly.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.domain.anomaly'`

- [ ] **Step 3: Implement `domain/anomaly.py`**

```python
"""Anomaly detection — flag invoices statistically unusual for their vendor.

Two signals, both relative to the vendor's own history: an *amount spike* (a total far
from the vendor's median, measured with a median/MAD robust z-score so one historical
outlier cannot mask the next) and *threshold gaming* (an amount parked just below a
round approval limit). The detector is pure robust statistics — no ML dependency — and
returns the dominant feature and a human-readable reason with every flag.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from apverify.domain.invoice import Invoice

_ROUND_MANTISSAS = (1, 2, 5)


class AnomalySeverity(Enum):
    NONE = "none"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class AnomalyFeatures:
    amount_robust_z: float
    threshold_proximity: float
    history_size: int


@dataclass(frozen=True, slots=True)
class AnomalyAssessment:
    score: float
    severity: AnomalySeverity
    top_feature: str
    reason: str


def extract_features(
    invoice: Invoice, history: Sequence[Invoice], band: float = 0.05
) -> AnomalyFeatures:
    amount = float(invoice.total.amount)
    amounts = [float(prior.total.amount) for prior in history]
    if not amounts:
        return AnomalyFeatures(0.0, _threshold_proximity(amount, band), 0)
    median = statistics.median(amounts)
    mad = statistics.median([abs(value - median) for value in amounts])
    scale = max(mad, abs(median) * 0.05, 1.0)  # floor: never divide by zero, damp tiny spreads
    return AnomalyFeatures(
        amount_robust_z=abs(amount - median) / scale,
        threshold_proximity=_threshold_proximity(amount, band),
        history_size=len(amounts),
    )


@dataclass(frozen=True, slots=True)
class RobustAnomalyDetector:
    min_history: int = 3
    sensitivity: float = 3.0
    band: float = 0.05
    high: float = 0.8
    medium: float = 0.5

    def score(self, invoice: Invoice, history: Sequence[Invoice]) -> AnomalyAssessment:
        features = extract_features(invoice, history, self.band)
        if features.history_size < self.min_history:
            return AnomalyAssessment(0.0, AnomalySeverity.NONE, "history", "insufficient history")
        spike = 1.0 - math.exp(-features.amount_robust_z / self.sensitivity)
        score = max(spike, features.threshold_proximity)
        top_feature = "amount_spike" if spike >= features.threshold_proximity else "threshold_gaming"
        severity = (
            AnomalySeverity.HIGH
            if score >= self.high
            else AnomalySeverity.MEDIUM
            if score >= self.medium
            else AnomalySeverity.NONE
        )
        return AnomalyAssessment(
            round(score, 4), severity, top_feature, _reason(top_feature, invoice, history, features)
        )


def _threshold_proximity(amount: float, band: float) -> float:
    if amount <= 0:
        return 0.0
    limit = _round_above(amount)
    gap = (limit - amount) / limit
    return 0.0 if gap >= band else 1.0 - gap / band


def _round_above(amount: float) -> float:
    exponent = math.floor(math.log10(amount))
    for mantissa in _ROUND_MANTISSAS:
        candidate = mantissa * 10**exponent
        if candidate > amount:
            return float(candidate)
    return float(10 ** (exponent + 1))


def _reason(
    top_feature: str, invoice: Invoice, history: Sequence[Invoice], features: AnomalyFeatures
) -> str:
    amount = float(invoice.total.amount)
    if top_feature == "threshold_gaming":
        return f"amount {amount:.0f} sits just under the {_round_above(amount):.0f} approval limit"
    median = statistics.median([float(prior.total.amount) for prior in history])
    multiple = amount / median if median else 0.0
    return (
        f"amount {amount:.0f} is {multiple:.1f}x the vendor's median {median:.0f} "
        f"(robust-z {features.amount_robust_z:.1f})"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/domain/test_anomaly.py -v`
Expected: PASS (all six).

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/domain/anomaly.py tests/unit/domain/test_anomaly.py
git commit -m "feat(anomaly): pure robust-statistics detector"
```

---

### Task 2: `reconcile_with_anomaly` in `domain/approval.py`

**Files:**
- Modify: `src/apverify/domain/approval.py`
- Test: `tests/unit/domain/test_approval.py`

**Interfaces:**
- Consumes: `FinalDecision`, `ApprovalDecision`, `_SEVERITY`, `Policy`/`DEFAULT_POLICY`, `apverify.domain.anomaly.{AnomalyAssessment, AnomalySeverity}`
- Produces: `reconcile_with_anomaly(decision: FinalDecision, assessment: AnomalyAssessment, policy: Policy = DEFAULT_POLICY) -> FinalDecision`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/domain/test_approval.py`:

```python
from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.approval import reconcile_with_anomaly


def test_high_anomaly_holds_the_payment() -> None:
    assessment = AnomalyAssessment(0.95, AnomalySeverity.HIGH, "amount_spike", "11x median")
    result = reconcile_with_anomaly(_approved(), assessment)
    assert result.decision is ApprovalDecision.HOLD
    assert any("11x median" in reason for reason in result.reasons)


def test_medium_anomaly_routes_to_human_review() -> None:
    assessment = AnomalyAssessment(0.6, AnomalySeverity.MEDIUM, "amount_spike", "elevated")
    result = reconcile_with_anomaly(_approved(), assessment)
    assert result.decision is ApprovalDecision.HUMAN_REVIEW


def test_no_anomaly_is_a_no_op() -> None:
    assessment = AnomalyAssessment(0.1, AnomalySeverity.NONE, "amount_spike", "normal")
    result = reconcile_with_anomaly(_approved(), assessment)
    assert result == _approved()
```

(`_approved()` already exists in this file from the BEC slice; reuse it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/domain/test_approval.py -k anomaly -v`
Expected: FAIL with `ImportError: cannot import name 'reconcile_with_anomaly'`

- [ ] **Step 3: Implement**

Add the import in `src/apverify/domain/approval.py` (with the other domain imports):

```python
from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
```

Add the function (after `reconcile_with_vendor_risk`):

```python
def reconcile_with_anomaly(
    decision: FinalDecision,
    assessment: AnomalyAssessment,
    policy: Policy = DEFAULT_POLICY,
) -> FinalDecision:
    """Fold a statistical-anomaly assessment into a decision: a HIGH anomaly holds the
    payment, a MEDIUM one routes to a human, and NONE is left untouched. Never lowers a
    decision."""
    if assessment.severity is AnomalySeverity.NONE:
        return decision
    target = (
        ApprovalDecision.HOLD
        if assessment.severity is AnomalySeverity.HIGH
        else ApprovalDecision.HUMAN_REVIEW
    )
    escalated = max(decision.decision, target, key=lambda d: _SEVERITY[d])
    reason = f"anomaly {assessment.top_feature}: {assessment.reason}"
    return FinalDecision(decision=escalated, reasons=(*decision.reasons, reason))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/domain/test_approval.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/domain/approval.py tests/unit/domain/test_approval.py
git commit -m "feat(approval): reconcile statistical anomaly (HIGH holds, MEDIUM reviews)"
```

---

### Task 3: `AnomalyDetector` + `VendorHistoryRepository` ports

**Files:**
- Modify: `src/apverify/application/ports.py`
- Test: `tests/unit/application/test_ports.py`

**Interfaces:**
- Consumes: `apverify.domain.anomaly.AnomalyAssessment`, `apverify.domain.invoice.Invoice`
- Produces:
  - `AnomalyDetector` (runtime-checkable Protocol): `score(self, invoice: Invoice, history: Sequence[Invoice]) -> AnomalyAssessment`
  - `VendorHistoryRepository` (runtime-checkable Protocol): `history_for(self, invoice: Invoice) -> tuple[Invoice, ...]`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/application/test_ports.py`:

```python
from collections.abc import Sequence

from apverify.application.ports import AnomalyDetector, VendorHistoryRepository
from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.invoice import Invoice


def test_objects_satisfy_the_anomaly_ports() -> None:
    class _Detector:
        def score(self, invoice: Invoice, history: Sequence[Invoice]) -> AnomalyAssessment:
            return AnomalyAssessment(0.0, AnomalySeverity.NONE, "history", "n/a")

    class _History:
        def history_for(self, invoice: Invoice) -> tuple[Invoice, ...]:
            return ()

    assert isinstance(_Detector(), AnomalyDetector)
    assert isinstance(_History(), VendorHistoryRepository)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/application/test_ports.py -k anomaly -v`
Expected: FAIL with `ImportError: cannot import name 'AnomalyDetector'`

- [ ] **Step 3: Implement**

Add the import in `src/apverify/application/ports.py` (with the other domain imports):

```python
from apverify.domain.anomaly import AnomalyAssessment
```

Add the ports (after `VendorMasterRepository`):

```python
@runtime_checkable
class AnomalyDetector(Protocol):
    """Scores how statistically unusual an invoice is for its vendor, given history."""

    def score(self, invoice: Invoice, history: Sequence[Invoice]) -> AnomalyAssessment: ...


@runtime_checkable
class VendorHistoryRepository(Protocol):
    """A vendor's previously-seen invoices, the baseline an anomaly is measured against."""

    def history_for(self, invoice: Invoice) -> tuple[Invoice, ...]: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/application/test_ports.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/application/ports.py tests/unit/application/test_ports.py
git commit -m "feat(ports): AnomalyDetector + VendorHistoryRepository"
```

---

### Task 4: In-memory vendor history + JSON loader

**Files:**
- Create: `src/apverify/infrastructure/anomaly/__init__.py` (empty)
- Create: `src/apverify/infrastructure/anomaly/history.py`
- Test: `tests/unit/infrastructure/test_vendor_history.py`

**Interfaces:**
- Consumes: `apverify.domain.invoice.Invoice`, `apverify.infrastructure.mapping.InvoiceDTO`, `apverify.infrastructure.errors.AdapterError`
- Produces:
  - `InMemoryVendorHistory(invoices: Sequence[Invoice])` with `history_for(invoice) -> tuple[Invoice, ...]`
  - `load_vendor_history(path: Path) -> InMemoryVendorHistory`
  - `VendorHistoryError(AdapterError)`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/infrastructure/test_vendor_history.py`:

```python
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from apverify.application.ports import VendorHistoryRepository
from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.value_objects import Money
from apverify.infrastructure.anomaly.history import (
    InMemoryVendorHistory,
    VendorHistoryError,
    load_vendor_history,
)


def _invoice(vendor: str, total: str) -> Invoice:
    amount = Money(Decimal(total))
    return Invoice(
        vendor_name=vendor,
        invoice_number="H",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=amount,
        tax=Money(Decimal("0")),
        total=amount,
        line_items=(LineItem("x", 1, amount, amount),),
        tax_breakdown=TaxBreakdown(),
        vendor_gstin="",
        purchase_order_ref="",
    )


def test_history_is_grouped_by_vendor() -> None:
    history = InMemoryVendorHistory(
        [_invoice("ACME", "100"), _invoice("ACME", "110"), _invoice("Other", "5")]
    )
    assert isinstance(history, VendorHistoryRepository)
    matched = history.history_for(_invoice("ACME", "999"))
    assert {str(inv.total.amount) for inv in matched} == {"100.00", "110.00"}


def test_load_vendor_history_parses_a_file(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "invoices": [
                    {
                        "vendor_name": "ACME",
                        "invoice_number": "H1",
                        "invoice_date": "04-06-2025",
                        "currency": "INR",
                        "subtotal": "100",
                        "tax": "0",
                        "total": "100",
                        "line_items": [],
                    }
                ]
            }
        )
    )
    history = load_vendor_history(path)
    assert len(history.history_for(_invoice("ACME", "1"))) == 1


def test_load_vendor_history_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VendorHistoryError):
        load_vendor_history(tmp_path / "nope.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/infrastructure/test_vendor_history.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.infrastructure.anomaly'`

- [ ] **Step 3: Implement**

Create `src/apverify/infrastructure/anomaly/__init__.py` (empty).

Create `src/apverify/infrastructure/anomaly/history.py`:

```python
"""Load a vendor's prior invoices — the baseline anomaly detection measures against."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from apverify.domain.invoice import Invoice
from apverify.infrastructure.errors import AdapterError
from apverify.infrastructure.mapping import InvoiceDTO, to_domain


class _HistoryFileDTO(BaseModel):
    invoices: list[InvoiceDTO]


class VendorHistoryError(AdapterError):
    """Vendor-history data could not be loaded."""


class InMemoryVendorHistory:
    def __init__(self, invoices: Sequence[Invoice]) -> None:
        self._by_vendor: dict[str, list[Invoice]] = {}
        for invoice in invoices:
            self._by_vendor.setdefault(invoice.vendor_name, []).append(invoice)

    def history_for(self, invoice: Invoice) -> tuple[Invoice, ...]:
        return tuple(self._by_vendor.get(invoice.vendor_name, ()))


def load_vendor_history(path: Path) -> InMemoryVendorHistory:
    try:
        document = _HistoryFileDTO.model_validate(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise VendorHistoryError(f"could not load vendor history from {path}: {exc}") from exc
    return InMemoryVendorHistory([to_domain(dto) for dto in document.invoices])
```

Note: confirm `InvoiceDTO` and `to_domain` are exported from `apverify.infrastructure.mapping` (they are used by the extractors). If `InvoiceDTO` requires fields the fixture omits, the test's JSON already supplies the core fields; adjust the fixture to match `InvoiceDTO`'s required fields if validation fails.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/infrastructure/test_vendor_history.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/infrastructure/anomaly/ tests/unit/infrastructure/test_vendor_history.py
git commit -m "feat(infra): in-memory vendor history + JSON loader"
```

---

### Task 5: Wire the anomaly step into `ReviewPayableUseCase`

**Files:**
- Modify: `src/apverify/application/review_payable.py`
- Test: `tests/unit/application/test_review_payable_anomaly.py`

**Interfaces:**
- Consumes: `apverify.application.ports.{AnomalyDetector, VendorHistoryRepository}`, `apverify.domain.anomaly.AnomalyAssessment`, `apverify.domain.approval.reconcile_with_anomaly`
- Produces: `ReviewPayableUseCase(..., anomaly_detector: AnomalyDetector | None = None, vendor_history: VendorHistoryRepository | None = None)`; `PayableReview.anomaly: AnomalyAssessment | None`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/application/test_review_payable_anomaly.py`:

```python
from __future__ import annotations

from pathlib import Path

from tests.support import (
    PO_NUMBER,
    FakeExtractor,
    FakeOcr,
    FakeRenderer,
    build_goods_receipt,
    build_invoice,
    build_purchase_order,
    build_raw_text,
)

from apverify.application.review_payable import ReviewPayableUseCase
from apverify.domain.anomaly import RobustAnomalyDetector
from apverify.domain.critique import ApprovalDecision
from apverify.domain.invoice import Invoice
from apverify.domain.value_objects import Money
from apverify.infrastructure.anomaly.history import InMemoryVendorHistory
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository

_DOC = Path("ignored-by-fakes.pdf")


def _history() -> InMemoryVendorHistory:
    return InMemoryVendorHistory(
        [build_invoice(total=Money.of(str(amount))) for amount in (90, 95, 100, 105, 110)]
    )


def _use_case(invoice: Invoice) -> ReviewPayableUseCase:
    return ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(invoice),
        ocr=FakeOcr(build_raw_text(invoice)),
        procurement=InMemoryProcurementRepository(
            purchase_orders=[build_purchase_order()],
            goods_receipts=[build_goods_receipt()],
        ),
        anomaly_detector=RobustAnomalyDetector(),
        vendor_history=_history(),
    )


def test_amount_spike_holds_the_payment() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER, total=Money.of("100000"))
    review = _use_case(invoice).execute(_DOC)
    assert review.decision.decision is ApprovalDecision.HOLD
    assert review.anomaly is not None
    assert any("anomaly" in reason for reason in review.decision.reasons)


def test_in_range_amount_stays_auto_approved() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER, total=Money.of("102"))
    review = _use_case(invoice).execute(_DOC)
    assert review.decision.decision is ApprovalDecision.AUTO_APPROVE
    assert review.anomaly is not None
```

Note: `build_invoice`'s default vendor is `"ACME Steel Pvt Ltd"`; `_history()` builds invoices with that same default vendor, so `history_for` matches. Confirm `build_raw_text` for the spike invoice still supports the OCR cross-check (the critic must not independently HOLD the in-range case — if it does, adjust the in-range total to one the critic accepts, keeping it within the history spread).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/application/test_review_payable_anomaly.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'anomaly_detector'`

- [ ] **Step 3: Implement**

In `src/apverify/application/review_payable.py`:

Add to the ports import list: `AnomalyDetector`, `VendorHistoryRepository`. Add to the approval import: `reconcile_with_anomaly`. Add a domain import:

```python
from apverify.domain.anomaly import AnomalyAssessment
```

Add the field to `PayableReview`:

```python
    anomaly: AnomalyAssessment | None = None
```

Add constructor params (after `vendor_master`) and store them:

```python
        anomaly_detector: AnomalyDetector | None = None,
        vendor_history: VendorHistoryRepository | None = None,
```
```python
        self._anomaly_detector = anomaly_detector
        self._vendor_history = vendor_history
```

In `execute`, after the vendor-risk block and before the `approve` trace record:

```python
        anomaly = self._anomaly(invoice, trace)
        if anomaly is not None:
            decision = reconcile_with_anomaly(decision, anomaly, self._critic_policy)
```

Pass it into `PayableReview(...)`:

```python
            anomaly=anomaly,
```

Add the step method (next to `_vendor_risk`):

```python
    def _anomaly(self, invoice: Invoice, trace: list[TraceEntry]) -> AnomalyAssessment | None:
        """Score the invoice against the vendor's history. Skipped unless both a
        detector and a history source are configured, leaving the pipeline unchanged."""
        if self._anomaly_detector is None or self._vendor_history is None:
            return None
        detector, history = self._anomaly_detector, self._vendor_history
        assessment, elapsed = self._timed(
            lambda: detector.score(invoice, history.history_for(invoice))
        )
        self._record(
            trace, "anomaly", f"{assessment.severity.value} ({assessment.top_feature})", elapsed
        )
        return assessment
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/application/test_review_payable_anomaly.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/application/review_payable.py tests/unit/application/test_review_payable_anomaly.py
git commit -m "feat(review): anomaly step holds on a statistical spike"
```

---

### Task 6: Settings + bootstrap wiring

**Files:**
- Modify: `src/apverify/infrastructure/settings.py`
- Modify: `src/apverify/interface/cli/bootstrap.py`
- Test: `tests/unit/interface/test_bootstrap.py`

**Interfaces:**
- Consumes: `apverify.infrastructure.anomaly.history.load_vendor_history`, `apverify.domain.anomaly.RobustAnomalyDetector`, `Settings.anomaly_history_path`
- Produces: `build_review_use_case(..., anomaly_detector=None, vendor_history=None)` wires the pure detector + history from `settings.anomaly_history_path` when set; `_build_vendor_history(settings) -> VendorHistoryRepository | None`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/interface/test_bootstrap.py`:

```python
from apverify.interface.cli.bootstrap import _build_vendor_history


def test_vendor_history_is_none_when_path_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("ANOMALY_HISTORY_PATH", "")
    assert _build_vendor_history(Settings()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/interface/test_bootstrap.py -k vendor_history -v`
Expected: FAIL with `ImportError: cannot import name '_build_vendor_history'`

- [ ] **Step 3: Implement**

In `src/apverify/infrastructure/settings.py`, add with the other optional paths:

```python
    anomaly_history_path: str = Field("", alias="ANOMALY_HISTORY_PATH")
```

In `src/apverify/interface/cli/bootstrap.py`, add imports:

```python
from apverify.application.ports import VendorHistoryRepository
from apverify.domain.anomaly import RobustAnomalyDetector
from apverify.infrastructure.anomaly.history import load_vendor_history
```

(Add `VendorHistoryRepository` to the existing `apverify.application.ports` import block instead of a second import if cleaner.)

Add params to `build_review_use_case` and pass them through:

```python
    anomaly_detector=anomaly_detector or (RobustAnomalyDetector() if _build_vendor_history(settings) else None),
    vendor_history=vendor_history or _build_vendor_history(settings),
```

with the new keyword parameters added to the signature:

```python
    anomaly_detector: AnomalyDetector | None = None,
    vendor_history: VendorHistoryRepository | None = None,
```

(import `AnomalyDetector` too) and the helper:

```python
def _build_vendor_history(settings: Settings) -> VendorHistoryRepository | None:
    if not settings.anomaly_history_path:
        return None
    return load_vendor_history(Path(settings.anomaly_history_path))
```

To avoid building history twice, compute it once:

```python
    history = vendor_history or _build_vendor_history(settings)
    return ReviewPayableUseCase(
        ...
        vendor_master=vendor_master or _build_vendor_master(settings),
        anomaly_detector=anomaly_detector or (RobustAnomalyDetector() if history else None),
        vendor_history=history,
        tracer=_build_tracer(settings),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/interface/test_bootstrap.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/infrastructure/settings.py src/apverify/interface/cli/bootstrap.py tests/unit/interface/test_bootstrap.py
git commit -m "feat(bootstrap): wire anomaly detector from ANOMALY_HISTORY_PATH"
```

---

### Task 7: Anomaly synthesis — `eval/anomaly_synthesis.py`

**Files:**
- Create: `src/apverify/eval/anomaly_synthesis.py`
- Test: `tests/unit/eval/test_anomaly_synthesis.py`

**Interfaces:**
- Consumes: `apverify.eval.synthetic.{GroundTruth, generate_dataset}`, `apverify.domain.invoice.Invoice`, `apverify.domain.value_objects.Money`
- Produces:
  - `AnomalyCase(invoice: Invoice, history: tuple[Invoice, ...], is_anomaly: bool, kind: str)` (frozen)
  - `build_anomaly_cases(base: Sequence[GroundTruth]) -> list[AnomalyCase]`
  - scenario constants `AMOUNT_SPIKE`, `THRESHOLD_GAMING`, `NORMAL`; `ANOMALY_KINDS = (AMOUNT_SPIKE, THRESHOLD_GAMING)`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_anomaly_synthesis.py`:

```python
from __future__ import annotations

from apverify.eval.anomaly_synthesis import ANOMALY_KINDS, SCENARIOS, build_anomaly_cases
from apverify.eval.synthetic import generate_dataset


def test_each_base_invoice_yields_one_case_per_scenario() -> None:
    cases = build_anomaly_cases(generate_dataset(count=3))
    assert {case.kind for case in cases} == set(SCENARIOS)
    assert all(len(case.history) >= 3 for case in cases)  # enough history to score


def test_anomaly_kinds_are_labelled_anomalous_and_normal_is_not() -> None:
    for case in build_anomaly_cases(generate_dataset(count=3)):
        assert case.is_anomaly == (case.kind in ANOMALY_KINDS)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_anomaly_synthesis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.eval.anomaly_synthesis'`

- [ ] **Step 3: Implement `eval/anomaly_synthesis.py`**

```python
"""Synthesise a labelled anomaly benchmark over base invoices.

For each base vendor we build a plausible history clustered around a median, then emit
the anomalies (an amount spike far above the median; an amount parked just under a round
approval limit) and the hard negative (an amount within the vendor's usual spread). The
normal case is what makes the false-positive number mean something.

Deterministic: fixed transforms of the base, no randomness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from apverify.domain.invoice import Invoice
from apverify.domain.value_objects import Money
from apverify.eval.synthetic import GroundTruth

AMOUNT_SPIKE = "amount_spike"
THRESHOLD_GAMING = "threshold_gaming"
NORMAL = "normal"

SCENARIOS = (AMOUNT_SPIKE, THRESHOLD_GAMING, NORMAL)
ANOMALY_KINDS = (AMOUNT_SPIKE, THRESHOLD_GAMING)

# A deterministic ±spread around the vendor's median, in multiples of the base total.
_HISTORY_SPREAD = (Decimal("0.90"), Decimal("0.95"), Decimal("1.00"), Decimal("1.05"), Decimal("1.10"))


@dataclass(frozen=True, slots=True)
class AnomalyCase:
    invoice: Invoice
    history: tuple[Invoice, ...]
    is_anomaly: bool
    kind: str


def build_anomaly_cases(base: Sequence[GroundTruth]) -> list[AnomalyCase]:
    cases: list[AnomalyCase] = []
    for truth in base:
        invoice = truth.invoice
        median = invoice.total.amount
        history = tuple(
            replace(invoice, total=Money(median * factor), subtotal=Money(median * factor))
            for factor in _HISTORY_SPREAD
        )
        cases.extend(
            [
                AnomalyCase(_at(invoice, median * 10), history, True, AMOUNT_SPIKE),
                AnomalyCase(_at(invoice, _just_under_round(median)), history, True, THRESHOLD_GAMING),
                AnomalyCase(_at(invoice, median * Decimal("1.02")), history, False, NORMAL),
            ]
        )
    return cases


def _at(invoice: Invoice, total: Decimal) -> Invoice:
    money = Money(total)
    return replace(invoice, total=money, subtotal=money)


def _just_under_round(median: Decimal) -> Decimal:
    """A value 0.5% below the next round number above the median — plausible for the
    vendor (small robust-z) yet parked under an approval limit."""
    from apverify.domain.anomaly import _round_above

    limit = Decimal(str(_round_above(float(median))))
    return (limit * Decimal("0.995")).quantize(Decimal("1"))
```

Note: importing the private `_round_above` keeps the gaming amount consistent with the detector's own notion of a round limit. If you prefer not to import a private helper, replicate the round-above computation here; consistency with the detector is the requirement.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/eval/test_anomaly_synthesis.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/anomaly_synthesis.py tests/unit/eval/test_anomaly_synthesis.py
git commit -m "feat(eval): synthetic anomaly injection (spike + threshold gaming)"
```

---

### Task 8: Anomaly metrics — `eval/anomaly_eval.py`

**Files:**
- Create: `src/apverify/eval/anomaly_eval.py`
- Test: `tests/unit/eval/test_anomaly_eval.py`

**Interfaces:**
- Consumes: `apverify.domain.anomaly.{RobustAnomalyDetector, AnomalySeverity}`, `apverify.application.ports.AnomalyDetector`, `apverify.eval.anomaly_synthesis.AnomalyCase`, `apverify.eval.fusion.auroc`
- Produces:
  - `DetectorResult(name: str, auroc: float, catch_rate: float, false_positive_rate: float)` (frozen)
  - `AnomalyReport(case_count: int, anomaly_count: int, results: tuple[DetectorResult, ...], sklearn_available: bool)` (frozen)
  - `evaluate_anomaly(cases: Sequence[AnomalyCase]) -> AnomalyReport`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_anomaly_eval.py`:

```python
from __future__ import annotations

from apverify.eval.anomaly_eval import evaluate_anomaly
from apverify.eval.anomaly_synthesis import build_anomaly_cases
from apverify.eval.synthetic import generate_dataset


def test_pure_detector_separates_anomalies_with_no_false_positives() -> None:
    report = evaluate_anomaly(build_anomaly_cases(generate_dataset(count=5)))
    pure = next(r for r in report.results if r.name == "robust-statistics")
    assert pure.catch_rate >= 0.9
    assert pure.false_positive_rate == 0.0
    assert pure.auroc >= 0.9


def test_report_always_includes_the_pure_detector() -> None:
    report = evaluate_anomaly(build_anomaly_cases(generate_dataset(count=5)))
    assert any(r.name == "robust-statistics" for r in report.results)


def test_empty_cases_yield_a_zeroed_report() -> None:
    report = evaluate_anomaly([])
    assert report.case_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_anomaly_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.eval.anomaly_eval'`

- [ ] **Step 3: Implement `eval/anomaly_eval.py`**

```python
"""Score anomaly detectors against the labelled benchmark — pure vs Isolation Forest.

The pure robust-statistics detector is always evaluated. If scikit-learn is installed,
the Isolation Forest detector is added for a head-to-head AUROC; if not, the benchmark
reports the pure detector alone and flags that sklearn was unavailable. The ML model
earns its dependency only if it wins.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.application.ports import AnomalyDetector
from apverify.domain.anomaly import AnomalySeverity, RobustAnomalyDetector
from apverify.eval.anomaly_synthesis import AnomalyCase
from apverify.eval.fusion import auroc


@dataclass(frozen=True, slots=True)
class DetectorResult:
    name: str
    auroc: float
    catch_rate: float
    false_positive_rate: float


@dataclass(frozen=True, slots=True)
class AnomalyReport:
    case_count: int
    anomaly_count: int
    results: tuple[DetectorResult, ...]
    sklearn_available: bool


def evaluate_anomaly(cases: Sequence[AnomalyCase]) -> AnomalyReport:
    detectors: list[tuple[str, AnomalyDetector]] = [("robust-statistics", RobustAnomalyDetector())]
    sklearn_available = False
    forest = _isolation_forest()
    if forest is not None:
        detectors.append(("isolation-forest", forest))
        sklearn_available = True

    results = tuple(_score(name, detector, cases) for name, detector in detectors)
    return AnomalyReport(
        case_count=len(cases),
        anomaly_count=sum(1 for case in cases if case.is_anomaly),
        results=results,
        sklearn_available=sklearn_available,
    )


def _score(name: str, detector: AnomalyDetector, cases: Sequence[AnomalyCase]) -> DetectorResult:
    scored = [
        (detector.score(case.invoice, case.history), case.is_anomaly) for case in cases
    ]
    samples = [(assessment.score, is_anomaly) for assessment, is_anomaly in scored]
    flagged = [
        (assessment.severity is not AnomalySeverity.NONE, is_anomaly)
        for assessment, is_anomaly in scored
    ]
    anomalies = [f for f, is_anomaly in flagged if is_anomaly]
    normals = [f for f, is_anomaly in flagged if not is_anomaly]
    return DetectorResult(
        name=name,
        auroc=auroc(samples),
        catch_rate=_rate([f for f in anomalies if f], anomalies),
        false_positive_rate=_rate([f for f in normals if f], normals),
    )


def _rate(hits: Sequence[object], population: Sequence[object]) -> float:
    return len(hits) / len(population) if population else 0.0


def _isolation_forest() -> AnomalyDetector | None:
    try:
        from apverify.infrastructure.anomaly.isolation_forest import IsolationForestDetector
    except ImportError:
        return None
    return IsolationForestDetector()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/eval/test_anomaly_eval.py -v`
Expected: PASS (Isolation Forest path simply absent until Task 10).

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/anomaly_eval.py tests/unit/eval/test_anomaly_eval.py
git commit -m "feat(eval): anomaly benchmark (pure detector, sklearn-optional)"
```

---

### Task 9: `render_anomaly` + `apverify-eval-anomaly` CLI

**Files:**
- Modify: `src/apverify/eval/report.py`
- Create: `src/apverify/eval/anomaly_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/eval/test_report.py` (append), `tests/unit/eval/test_anomaly_cli.py`

**Interfaces:**
- Consumes: `apverify.eval.anomaly_eval.{AnomalyReport, DetectorResult}`, `evaluate_anomaly`, `build_anomaly_cases`, `generate_dataset`
- Produces: `render_anomaly(report: AnomalyReport, console: Console | None = None) -> None`; `apverify-eval-anomaly` Typer app with `--count`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/eval/test_report.py`:

```python
from apverify.eval.anomaly_eval import AnomalyReport, DetectorResult
from apverify.eval.report import render_anomaly


def test_render_anomaly_prints_each_detector() -> None:
    report = AnomalyReport(
        case_count=15,
        anomaly_count=10,
        results=(DetectorResult("robust-statistics", 0.97, 1.0, 0.0),),
        sklearn_available=False,
    )
    console = Console(record=True, width=100)
    render_anomaly(report, console)
    text = console.export_text()
    assert "robust-statistics" in text
    assert "anomaly" in text.lower()
```

Create `tests/unit/eval/test_anomaly_cli.py`:

```python
from __future__ import annotations

from typer.testing import CliRunner

from apverify.eval.anomaly_cli import app


def test_cli_runs_the_anomaly_benchmark() -> None:
    result = CliRunner().invoke(app, ["--count", "5"])
    assert result.exit_code == 0
    assert "anomaly" in result.stdout.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_report.py -k render_anomaly tests/unit/eval/test_anomaly_cli.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement**

Add the import in `src/apverify/eval/report.py` (with the eval imports):

```python
from apverify.eval.anomaly_eval import AnomalyReport
```

Add the function (near `render_fraud`):

```python
def render_anomaly(report: AnomalyReport, console: Console | None = None) -> None:
    console = console or Console()
    if report.case_count == 0:
        console.print("[yellow]No anomaly cases to evaluate.[/yellow]")
        return

    console.print(
        f"[bold]Anomaly detection[/bold] (n={report.case_count}, "
        f"{report.anomaly_count} anomalous)."
        + ("" if report.sklearn_available else " [dim]scikit-learn not installed — "
           "pure detector only.[/dim]")
    )
    table = Table(title="Detector comparison", title_justify="left")
    table.add_column("Detector", style="cyan")
    table.add_column("AUROC", justify="right")
    table.add_column("Caught", justify="right")
    table.add_column("False-pos", justify="right")
    for result in report.results:
        table.add_row(
            result.name,
            f"{result.auroc:.3f}",
            f"{result.catch_rate:.0%}",
            f"{result.false_positive_rate:.0%}",
        )
    console.print(table)
```

Create `src/apverify/eval/anomaly_cli.py`:

```python
"""``apverify-eval-anomaly`` — anomaly-detection benchmark (pure vs Isolation Forest).

Synthetic only. Isolation Forest is included when scikit-learn is installed
(``pip install -e '.[anomaly]'``); otherwise the pure detector is reported alone.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.anomaly_eval import evaluate_anomaly
from apverify.eval.anomaly_synthesis import build_anomaly_cases
from apverify.eval.report import render_anomaly
from apverify.eval.synthetic import generate_dataset

app = typer.Typer(add_completion=False, help="Anomaly-detection benchmark.")


@app.command()
def run(count: Annotated[int, typer.Option(help="Synthetic base invoices.")] = 25) -> None:
    report = evaluate_anomaly(build_anomaly_cases(generate_dataset(count=count)))
    render_anomaly(report, Console())
```

In `pyproject.toml`, add the optional extra (after `langfuse = [...]`):

```toml
anomaly = ["scikit-learn>=1.5"]
```

and the console script (with the other `apverify-eval-*`):

```toml
apverify-eval-anomaly = "apverify.eval.anomaly_cli:app"
```

Reinstall: `pip install -e .`

- [ ] **Step 4: Run tests + smoke-run**

Run: `pytest tests/unit/eval/test_report.py tests/unit/eval/test_anomaly_cli.py -v`
Expected: PASS.
Then: `apverify-eval-anomaly --count 25`
Expected: a detector-comparison table with `robust-statistics` AUROC ≥0.9, caught 100%, false-pos 0% (and a "scikit-learn not installed" note until Task 10).

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/report.py src/apverify/eval/anomaly_cli.py pyproject.toml tests/unit/eval/test_report.py tests/unit/eval/test_anomaly_cli.py
git commit -m "feat(cli): apverify-eval-anomaly benchmark"
```

---

### Task 10: Isolation Forest challenger (optional sklearn adapter)

**Files:**
- Create: `src/apverify/infrastructure/anomaly/isolation_forest.py`
- Test: `tests/unit/infrastructure/test_isolation_forest.py`

**Interfaces:**
- Consumes: `apverify.domain.anomaly.{AnomalyAssessment, AnomalySeverity}`, `apverify.domain.invoice.Invoice`, `scikit-learn`
- Produces: `IsolationForestDetector(min_history: int = 3)` with `score(invoice, history) -> AnomalyAssessment` (satisfies `AnomalyDetector`)

- [ ] **Step 1: Install the optional extra**

Run: `pip install -e '.[anomaly]'`
Expected: scikit-learn installed.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/infrastructure/test_isolation_forest.py`:

```python
from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("sklearn")

from apverify.domain.anomaly import AnomalySeverity  # noqa: E402
from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown  # noqa: E402
from apverify.domain.value_objects import Money  # noqa: E402
from apverify.infrastructure.anomaly.isolation_forest import (  # noqa: E402
    IsolationForestDetector,
)


def _invoice(total: str) -> Invoice:
    amount = Money(Decimal(total))
    return Invoice(
        vendor_name="ACME",
        invoice_number="I",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=amount,
        tax=Money(Decimal("0")),
        total=amount,
        line_items=(LineItem("x", 1, amount, amount),),
        tax_breakdown=TaxBreakdown(),
        vendor_gstin="",
        purchase_order_ref="",
    )


def test_isolation_forest_scores_a_spike_above_a_normal() -> None:
    history = [_invoice(str(t)) for t in (90, 95, 100, 105, 110)]
    detector = IsolationForestDetector()
    spike = detector.score(_invoice("1000"), history)
    normal = detector.score(_invoice("102"), history)
    assert spike.score > normal.score


def test_insufficient_history_abstains() -> None:
    detector = IsolationForestDetector()
    result = detector.score(_invoice("1000"), [_invoice("100")])
    assert result.severity is AnomalySeverity.NONE
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/infrastructure/test_isolation_forest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.infrastructure.anomaly.isolation_forest'`

- [ ] **Step 4: Implement**

Create `src/apverify/infrastructure/anomaly/isolation_forest.py`:

```python
"""Isolation Forest anomaly detector — the optional ML challenger to the pure baseline.

Fits an Isolation Forest on the vendor's historical totals and scores the candidate's
total against it. It sees only the amount (not the threshold-proximity feature the pure
detector engineers), so the benchmark shows where domain knowledge beats a generic ML
model. Lives behind the optional ``anomaly`` extra and is used only by the eval harness.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from sklearn.ensemble import IsolationForest

from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.invoice import Invoice


class IsolationForestDetector:
    def __init__(self, min_history: int = 3, high: float = 0.8, medium: float = 0.5) -> None:
        self._min_history = min_history
        self._high = high
        self._medium = medium

    def score(self, invoice: Invoice, history: Sequence[Invoice]) -> AnomalyAssessment:
        amounts = [[float(prior.total.amount)] for prior in history]
        if len(amounts) < self._min_history:
            return AnomalyAssessment(0.0, AnomalySeverity.NONE, "history", "insufficient history")
        model = IsolationForest(random_state=0, n_estimators=100).fit(amounts)
        decision = float(model.decision_function([[float(invoice.total.amount)]])[0])
        score = 1.0 / (1.0 + math.exp(decision))  # lower decision ⇒ more anomalous ⇒ higher score
        severity = (
            AnomalySeverity.HIGH
            if score >= self._high
            else AnomalySeverity.MEDIUM
            if score >= self._medium
            else AnomalySeverity.NONE
        )
        return AnomalyAssessment(
            round(score, 4), severity, "isolation_forest", "isolation-forest amount outlier score"
        )
```

- [ ] **Step 5: Run test + head-to-head**

Run: `pytest tests/unit/infrastructure/test_isolation_forest.py -v`
Expected: PASS.
Then: `apverify-eval-anomaly --count 25`
Expected: now two rows — `robust-statistics` and `isolation-forest`. Note in the output whether the pure detector beats Isolation Forest on `threshold_gaming` (it should, since IF never sees the threshold feature) — this is the honest head-to-head finding.

- [ ] **Step 6: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`
(mypy: scikit-learn ships no stubs; if `import-untyped` is raised, add `[[tool.mypy.overrides]]` for `sklearn.*` with `ignore_missing_imports = true` in `pyproject.toml`, matching how other untyped third-party libs are handled.)

```bash
git add src/apverify/infrastructure/anomaly/isolation_forest.py tests/unit/infrastructure/test_isolation_forest.py pyproject.toml
git commit -m "feat(anomaly): optional Isolation Forest challenger"
```

---

## Final verification (after all tasks)

- [ ] `ruff check . && ruff format --check . && mypy --strict src tests` — clean
- [ ] `pytest -q` — all pass; domain layer 100% (`pytest --cov=apverify.domain --cov-report=term-missing`)
- [ ] `apverify-eval-anomaly --count 25` — robust-statistics AUROC ≥0.9, caught ≥90% @ 0% FP; Isolation Forest row present (sklearn installed), with the honest per-anomaly comparison
- [ ] Pipeline HOLD on a spike verified by Task 5's test
- [ ] README: add an anomaly-detection paragraph + the CLI + the `[anomaly]` extra (small follow-up)

## Spec coverage check

- Pure baseline + optional sklearn head-to-head → Tasks 1, 8, 10 ✓
- Eval + pipeline wiring → Tasks 3–6 ✓
- Robust-z (MAD) + threshold-proximity + abstention → Task 1 ✓
- reconcile (HIGH→HOLD, MEDIUM→review) → Task 2 ✓
- AnomalyDetector + VendorHistoryRepository ports → Task 3 ✓
- Synthetic spike/gaming + normal hard negative → Task 7 ✓
- Head-to-head AUROC, sklearn-absent degrades cleanly → Tasks 8, 10 ✓
- XAI top-feature + reason → Task 1 ✓
- Live pipeline HOLD + audit trace → Tasks 5, 6 ✓
- Acceptance (≥90%@0%-FP, AUROC ≥0.9, sklearn comparison, pipeline HOLD, domain 100%) → Tasks 5, 8 + Final verification ✓
```
