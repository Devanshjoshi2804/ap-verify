# Vendor-master / bank-change / BEC detection (v5 slice 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect vendor-master / bank-change / impersonation (BEC) fraud with explainable severity-tiered flags, wired into the live approval pipeline and measured by a synthetic benchmark.

**Architecture:** A pure `domain/vendor_master.py` assessor classifies an invoice against a vendor master into CLEAN / NEW_PAYEE / BANK_CHANGE / IMPERSONATION with a severity and a name-similarity score. A `VendorMasterRepository` port + `JsonVendorMaster` adapter feed it; `ReviewPayableUseCase` runs it as an optional step and `reconcile_with_vendor_risk` escalates HIGH→HOLD. A synthetic BEC benchmark + `apverify-eval-bec` measures it.

**Tech Stack:** Python 3.12, frozen dataclasses, `difflib` (stdlib), pydantic v2 DTOs, typer/rich, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-20-v5-vendor-master-bec-design.md`

## Global Constraints

- Clean/hexagonal: `domain` imports only `domain`; `application` imports `domain`; `infrastructure`/`eval`/`interface` may import inward. (dependency rule)
- No new third-party dependencies. `difflib`, `json`, `re` are stdlib.
- Determinism: no `random`, no wall-clock in domain/eval; synthetic values derive from index.
- Name matching uses **`canonical` (non-folded)** comparison — folding confusables would hide a typo-squat (`Stee1`→`Steel`), defeating impersonation detection.
- Severity is a fixed function of kind: CLEAN→NONE, NEW_PAYEE→LOW, BANK_CHANGE→HIGH, IMPERSONATION→HIGH.
- HOLD is driven by HIGH severity only; LOW (new-payee) is surfaced as a reason but never changes the decision.
- Gates after every task: `ruff check .`, `ruff format --check .`, `mypy --strict src tests`, `pytest`. **Domain layer 100% coverage.**
- No AI-tells; match surrounding idiom.
- **Git note:** the working dir is not a git repo, so `git commit` steps are no-ops until `git init`. The real per-task checkpoint is the gate suite — run it where the commit step appears.

---

### Task 1: Domain vendor-master assessor

**Files:**
- Create: `src/apverify/domain/vendor_master.py`
- Test: `tests/unit/domain/test_vendor_master.py`

**Interfaces:**
- Consumes: `apverify.domain.invoice.Invoice`, `apverify.domain.ocr.canonical`
- Produces:
  - `class Severity(Enum)`: `NONE`, `LOW`, `HIGH`
  - `class VendorRiskKind(Enum)`: `CLEAN`, `NEW_PAYEE`, `BANK_CHANGE`, `IMPERSONATION`
  - `KnownVendor(name: str, bank_accounts: frozenset[str], gstin: str = "")` (frozen)
  - `VendorRiskAssessment(kind: VendorRiskKind, severity: Severity, score: float, matched_vendor: str, reason: str)` (frozen)
  - `assess_vendor_risk(invoice: Invoice, master: Sequence[KnownVendor], impersonation_threshold: float = 0.85) -> VendorRiskAssessment`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/domain/test_vendor_master.py`:

```python
from __future__ import annotations

from decimal import Decimal

from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.value_objects import Money
from apverify.domain.vendor_master import (
    KnownVendor,
    Severity,
    VendorRiskKind,
    assess_vendor_risk,
)

_MASTER = [
    KnownVendor("ACME Steel Pvt Ltd", frozenset({"ACCT-0001"})),
    KnownVendor("Bharat Textiles LLP", frozenset({"ACCT-0002"})),
]


def _invoice(*, vendor: str = "ACME Steel Pvt Ltd", bank: str | None = None) -> Invoice:
    amount = Money(Decimal("100"))
    return Invoice(
        vendor_name=vendor,
        invoice_number="INV-1",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=amount,
        tax=Money(Decimal("0")),
        total=amount,
        line_items=(LineItem("Widget", 1, amount, amount),),
        tax_breakdown=TaxBreakdown(),
        vendor_gstin="",
        bank_account=bank,
        purchase_order_ref="",
    )


def test_known_vendor_with_known_bank_is_clean() -> None:
    result = assess_vendor_risk(_invoice(bank="ACCT-0001"), _MASTER)
    assert result.kind is VendorRiskKind.CLEAN
    assert result.severity is Severity.NONE


def test_known_vendor_with_no_bank_on_invoice_is_clean() -> None:
    result = assess_vendor_risk(_invoice(bank=None), _MASTER)
    assert result.kind is VendorRiskKind.CLEAN


def test_known_vendor_with_changed_bank_is_high_bank_change() -> None:
    result = assess_vendor_risk(_invoice(bank="ACCT-9999"), _MASTER)
    assert result.kind is VendorRiskKind.BANK_CHANGE
    assert result.severity is Severity.HIGH
    assert "bank" in result.reason.lower()


def test_typosquatted_name_is_high_impersonation() -> None:
    # "ACME Stee1" — one confusable substitution of a known vendor.
    result = assess_vendor_risk(_invoice(vendor="ACME Stee1 Pvt Ltd", bank="ACCT-9999"), _MASTER)
    assert result.kind is VendorRiskKind.IMPERSONATION
    assert result.severity is Severity.HIGH
    assert result.matched_vendor == "ACME Steel Pvt Ltd"


def test_unrelated_vendor_is_low_new_payee() -> None:
    result = assess_vendor_risk(_invoice(vendor="Konkan Foods Pvt Ltd", bank="ACCT-7777"), _MASTER)
    assert result.kind is VendorRiskKind.NEW_PAYEE
    assert result.severity is Severity.LOW


def test_empty_master_is_new_payee() -> None:
    result = assess_vendor_risk(_invoice(), [])
    assert result.kind is VendorRiskKind.NEW_PAYEE
    assert result.score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/domain/test_vendor_master.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.domain.vendor_master'`

- [ ] **Step 3: Implement `domain/vendor_master.py`**

```python
"""Vendor-master / bank-change / BEC detection — the highest-loss v5 fraud signal.

Business-email-compromise redirects payment by changing a known vendor's bank account
at the last minute, or by impersonating a vendor with a typo-squatted name. This
assessor checks an invoice against a vendor master and returns a discrete kind + a
severity (the explainable flag) plus the name-similarity to the nearest known vendor
(the continuous score the benchmark sweeps).

Name matching is deliberately *not* confusable-folded: folding would collapse a
``Stee1`` typo-squat onto ``Steel`` and hide exactly the impersonation we are hunting.

Pure domain logic: no I/O, deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum

from apverify.domain.invoice import Invoice
from apverify.domain.ocr import canonical


class Severity(Enum):
    NONE = "none"
    LOW = "low"
    HIGH = "high"


class VendorRiskKind(Enum):
    CLEAN = "clean"
    NEW_PAYEE = "new_payee"
    BANK_CHANGE = "bank_change"
    IMPERSONATION = "impersonation"


_SEVERITY: dict[VendorRiskKind, Severity] = {
    VendorRiskKind.CLEAN: Severity.NONE,
    VendorRiskKind.NEW_PAYEE: Severity.LOW,
    VendorRiskKind.BANK_CHANGE: Severity.HIGH,
    VendorRiskKind.IMPERSONATION: Severity.HIGH,
}


@dataclass(frozen=True, slots=True)
class KnownVendor:
    name: str
    bank_accounts: frozenset[str]
    gstin: str = ""


@dataclass(frozen=True, slots=True)
class VendorRiskAssessment:
    kind: VendorRiskKind
    severity: Severity
    score: float
    matched_vendor: str
    reason: str


def assess_vendor_risk(
    invoice: Invoice,
    master: Sequence[KnownVendor],
    impersonation_threshold: float = 0.85,
) -> VendorRiskAssessment:
    if not master:
        return _assess(VendorRiskKind.NEW_PAYEE, 0.0, "", "no known vendors to match against")

    nearest = max(master, key=lambda vendor: _name_similarity(invoice.vendor_name, vendor.name))
    score = _name_similarity(invoice.vendor_name, nearest.name)

    if canonical(invoice.vendor_name) == canonical(nearest.name):
        if invoice.bank_account and not _bank_known(invoice.bank_account, nearest.bank_accounts):
            return _assess(
                VendorRiskKind.BANK_CHANGE,
                score,
                nearest.name,
                f"bank account on known vendor {nearest.name} changed: "
                f"{_mask(invoice.bank_account)} not among its known accounts",
            )
        return _assess(VendorRiskKind.CLEAN, score, nearest.name, f"known vendor {nearest.name}")

    if score >= impersonation_threshold:
        return _assess(
            VendorRiskKind.IMPERSONATION,
            score,
            nearest.name,
            f"vendor {invoice.vendor_name!r} is a {score:.2f} name-match to known "
            f"{nearest.name!r} but not identical — possible impersonation",
        )

    return _assess(
        VendorRiskKind.NEW_PAYEE,
        score,
        nearest.name,
        f"vendor {invoice.vendor_name!r} matches no known vendor",
    )


def _assess(kind: VendorRiskKind, score: float, matched: str, reason: str) -> VendorRiskAssessment:
    return VendorRiskAssessment(kind, _SEVERITY[kind], round(score, 4), matched, reason)


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, canonical(a), canonical(b)).ratio()


def _bank_known(account: str, known: frozenset[str]) -> bool:
    return _account(account) in {_account(k) for k in known}


def _account(value: str) -> str:
    return value.replace(" ", "").upper()


def _mask(account: str) -> str:
    digits = _account(account)
    return f"****{digits[-4:]}" if len(digits) >= 4 else digits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/domain/test_vendor_master.py -v`
Expected: PASS (all six).

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/domain/vendor_master.py tests/unit/domain/test_vendor_master.py
git commit -m "feat(vendor-master): BEC risk assessor with severity tiers"
```

---

### Task 2: `reconcile_with_vendor_risk` in `domain/approval.py`

**Files:**
- Modify: `src/apverify/domain/approval.py`
- Test: `tests/unit/domain/test_approval.py` (create if absent)

**Interfaces:**
- Consumes: `FinalDecision`, `ApprovalDecision`, `_SEVERITY`, `Policy`/`DEFAULT_POLICY`, `apverify.domain.vendor_master.{VendorRiskAssessment, Severity}`
- Produces: `reconcile_with_vendor_risk(decision: FinalDecision, assessment: VendorRiskAssessment, policy: Policy = DEFAULT_POLICY) -> FinalDecision`

- [ ] **Step 1: Write the failing tests**

Create/append `tests/unit/domain/test_approval.py`:

```python
from __future__ import annotations

from apverify.domain.approval import FinalDecision, reconcile_with_vendor_risk
from apverify.domain.critique import ApprovalDecision
from apverify.domain.vendor_master import Severity, VendorRiskAssessment, VendorRiskKind


def _decision() -> FinalDecision:
    return FinalDecision(decision=ApprovalDecision.AUTO_APPROVE, reasons=("clean",))


def test_high_vendor_risk_holds_the_payment() -> None:
    assessment = VendorRiskAssessment(
        VendorRiskKind.BANK_CHANGE, Severity.HIGH, 1.0, "ACME", "bank changed"
    )
    result = reconcile_with_vendor_risk(_decision(), assessment)
    assert result.decision is ApprovalDecision.HOLD
    assert any("bank changed" in reason for reason in result.reasons)


def test_low_vendor_risk_adds_a_reason_but_does_not_change_the_decision() -> None:
    assessment = VendorRiskAssessment(
        VendorRiskKind.NEW_PAYEE, Severity.LOW, 0.2, "", "new vendor"
    )
    result = reconcile_with_vendor_risk(_decision(), assessment)
    assert result.decision is ApprovalDecision.AUTO_APPROVE
    assert any("new vendor" in reason for reason in result.reasons)


def test_clean_vendor_risk_is_a_no_op() -> None:
    assessment = VendorRiskAssessment(VendorRiskKind.CLEAN, Severity.NONE, 1.0, "ACME", "known")
    result = reconcile_with_vendor_risk(_decision(), assessment)
    assert result == _decision()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/domain/test_approval.py -k vendor_risk -v`
Expected: FAIL with `ImportError: cannot import name 'reconcile_with_vendor_risk'`

- [ ] **Step 3: Implement**

Add to `src/apverify/domain/approval.py` the import (with the other domain imports):

```python
from apverify.domain.vendor_master import Severity, VendorRiskAssessment
```

And the function (place after `reconcile_with_consistency`):

```python
def reconcile_with_vendor_risk(
    decision: FinalDecision,
    assessment: VendorRiskAssessment,
    policy: Policy = DEFAULT_POLICY,
) -> FinalDecision:
    """Fold a vendor-master / BEC assessment into an existing decision.

    A HIGH-severity flag (a changed bank account or an impersonated vendor) is the
    redirected-payment worst case, so it holds. A LOW flag (a new payee — common and
    usually legitimate) is surfaced as a reason but never blocks. Never lowers a
    decision.
    """
    reason = f"vendor-risk {assessment.kind.value}: {assessment.reason}"
    if assessment.severity is Severity.HIGH:
        held = max(decision.decision, ApprovalDecision.HOLD, key=lambda d: _SEVERITY[d])
        return FinalDecision(decision=held, reasons=decision.reasons + (reason,))
    if assessment.severity is Severity.LOW:
        return FinalDecision(decision=decision.decision, reasons=decision.reasons + (reason,))
    return decision
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/domain/test_approval.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/domain/approval.py tests/unit/domain/test_approval.py
git commit -m "feat(approval): reconcile vendor-risk (HIGH holds, LOW informs)"
```

---

### Task 3: `VendorMasterRepository` port

**Files:**
- Modify: `src/apverify/application/ports.py`
- Test: `tests/unit/application/test_ports.py`

**Interfaces:**
- Consumes: `apverify.domain.vendor_master.KnownVendor`
- Produces: `VendorMasterRepository` (runtime-checkable Protocol) with `known_vendors(self) -> tuple[KnownVendor, ...]`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/application/test_ports.py`:

```python
from apverify.application.ports import VendorMasterRepository
from apverify.domain.vendor_master import KnownVendor


def test_a_simple_object_satisfies_the_vendor_master_port() -> None:
    class _Master:
        def known_vendors(self) -> tuple[KnownVendor, ...]:
            return ()

    assert isinstance(_Master(), VendorMasterRepository)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/application/test_ports.py -k vendor_master -v`
Expected: FAIL with `ImportError: cannot import name 'VendorMasterRepository'`

- [ ] **Step 3: Implement**

Add the import in `src/apverify/application/ports.py` (with the other domain imports):

```python
from apverify.domain.vendor_master import KnownVendor
```

And the port (after `InvoiceLedger`):

```python
@runtime_checkable
class VendorMasterRepository(Protocol):
    """The roster of known vendors and their established bank accounts, checked against
    an incoming invoice for bank-change / impersonation (BEC) risk."""

    def known_vendors(self) -> tuple[KnownVendor, ...]: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/application/test_ports.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/application/ports.py tests/unit/application/test_ports.py
git commit -m "feat(ports): VendorMasterRepository"
```

---

### Task 4: `JsonVendorMaster` adapter

**Files:**
- Create: `src/apverify/infrastructure/vendor_master/__init__.py` (empty)
- Create: `src/apverify/infrastructure/vendor_master/repository.py`
- Test: `tests/unit/infrastructure/test_vendor_master.py`

**Interfaces:**
- Consumes: `apverify.domain.vendor_master.KnownVendor`, `apverify.infrastructure.errors.AdapterError`
- Produces:
  - `InMemoryVendorMaster(vendors: Sequence[KnownVendor])` with `known_vendors() -> tuple[KnownVendor, ...]`
  - `load_vendor_master(path: Path) -> InMemoryVendorMaster`
  - `VendorMasterError(AdapterError)`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/infrastructure/test_vendor_master.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from apverify.application.ports import VendorMasterRepository
from apverify.infrastructure.vendor_master.repository import (
    VendorMasterError,
    load_vendor_master,
)


def test_load_vendor_master_parses_a_file(tmp_path: Path) -> None:
    path = tmp_path / "vendors.json"
    path.write_text(
        json.dumps(
            {"vendors": [{"name": "ACME Steel Pvt Ltd", "bank_accounts": ["ACCT-0001"]}]}
        )
    )
    master = load_vendor_master(path)
    assert isinstance(master, VendorMasterRepository)
    vendors = master.known_vendors()
    assert vendors[0].name == "ACME Steel Pvt Ltd"
    assert vendors[0].bank_accounts == frozenset({"ACCT-0001"})


def test_load_vendor_master_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VendorMasterError):
        load_vendor_master(tmp_path / "nope.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/infrastructure/test_vendor_master.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.infrastructure.vendor_master'`

- [ ] **Step 3: Implement**

Create `src/apverify/infrastructure/vendor_master/__init__.py` (empty file).

Create `src/apverify/infrastructure/vendor_master/repository.py`:

```python
"""Load the vendor master from a JSON file into an in-memory repository."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from apverify.domain.vendor_master import KnownVendor
from apverify.infrastructure.errors import AdapterError


class _KnownVendorDTO(BaseModel):
    name: str
    bank_accounts: list[str]
    gstin: str = ""


class _VendorMasterFileDTO(BaseModel):
    vendors: list[_KnownVendorDTO]


class VendorMasterError(AdapterError):
    """Vendor-master data could not be loaded."""


class InMemoryVendorMaster:
    def __init__(self, vendors: Sequence[KnownVendor]) -> None:
        self._vendors = tuple(vendors)

    def known_vendors(self) -> tuple[KnownVendor, ...]:
        return self._vendors


def load_vendor_master(path: Path) -> InMemoryVendorMaster:
    try:
        document = _VendorMasterFileDTO.model_validate(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise VendorMasterError(f"could not load vendor master from {path}: {exc}") from exc
    return InMemoryVendorMaster(
        [
            KnownVendor(dto.name, frozenset(dto.bank_accounts), dto.gstin)
            for dto in document.vendors
        ]
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/infrastructure/test_vendor_master.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/infrastructure/vendor_master/ tests/unit/infrastructure/test_vendor_master.py
git commit -m "feat(infra): JSON vendor-master adapter"
```

---

### Task 5: Wire the vendor-risk step into `ReviewPayableUseCase`

**Files:**
- Modify: `src/apverify/application/review_payable.py`
- Test: `tests/unit/application/test_review_payable.py` (append; create if absent)

**Interfaces:**
- Consumes: `apverify.application.ports.VendorMasterRepository`, `apverify.domain.vendor_master.{assess_vendor_risk, VendorRiskAssessment}`, `apverify.domain.approval.reconcile_with_vendor_risk`
- Produces: `ReviewPayableUseCase(..., vendor_master: VendorMasterRepository | None = None)`; `PayableReview.vendor_risk: VendorRiskAssessment | None`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/application/test_review_payable.py`:

```python
from pathlib import Path

from apverify.application.review_payable import ReviewPayableUseCase
from apverify.domain.critique import ApprovalDecision
from apverify.domain.vendor_master import KnownVendor


class _Master:
    def known_vendors(self) -> tuple[KnownVendor, ...]:
        return (KnownVendor("ACME Steel Pvt Ltd", frozenset({"ACCT-0001"})),)


def test_bank_change_holds_the_payment(review_dependencies) -> None:
    # review_dependencies is a fixture providing fakes whose extractor returns an
    # invoice for "ACME Steel Pvt Ltd" with bank_account "ACCT-9999" (an unknown bank).
    use_case = ReviewPayableUseCase(**review_dependencies, vendor_master=_Master())
    result = use_case.execute(Path("dummy.pdf"))
    assert result.decision.decision is ApprovalDecision.HOLD
    assert result.vendor_risk is not None
    assert any("vendor-risk" in reason for reason in result.decision.reasons)
```

Note: if `tests/unit/application/test_review_payable.py` and a `review_dependencies`
fixture do not already exist, create the fixture in this file with fakes for
`renderer` (returns one `PageImage`), `extractor` (returns the ACME invoice with
`bank_account="ACCT-9999"`), `ocr` (returns a `RawText` containing the invoice fields),
and `procurement` (an `InMemoryProcurementRepository()`), shaped to the
`ReviewPayableUseCase.__init__` signature. Reuse `tests/support` factories where they
exist.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/application/test_review_payable.py -k bank_change -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'vendor_master'`

- [ ] **Step 3: Implement**

In `src/apverify/application/review_payable.py`:

Add imports:

```python
from apverify.application.ports import (
    DocumentRenderer,
    InvoiceExtractor,
    OcrTextProvider,
    ProcurementRepository,
    SemanticAuditor,
    VendorMasterRepository,
)
from apverify.domain.approval import (
    FinalDecision,
    approve,
    reconcile_with_audit,
    reconcile_with_consistency,
    reconcile_with_vendor_risk,
)
from apverify.domain.vendor_master import VendorRiskAssessment, assess_vendor_risk
```

Add the field to `PayableReview`:

```python
    vendor_risk: VendorRiskAssessment | None = None
```

Add the constructor parameter (after `secondary_extractor`) and store it:

```python
        vendor_master: VendorMasterRepository | None = None,
```
```python
        self._vendor_master = vendor_master
```

In `execute`, after the consistency reconciliation block and before the `approve`
trace record, add:

```python
        vendor_risk = self._vendor_risk(invoice, trace)
        if vendor_risk is not None:
            decision = reconcile_with_vendor_risk(decision, vendor_risk, self._critic_policy)
```

Pass it into the returned `PayableReview(...)`:

```python
            vendor_risk=vendor_risk,
```

And add the step method (next to `_consistency`):

```python
    def _vendor_risk(
        self, invoice: Invoice, trace: list[TraceEntry]
    ) -> VendorRiskAssessment | None:
        """Check the invoice's vendor + bank against the master for BEC risk. Absent a
        master the step is skipped, leaving the pipeline unchanged."""
        if self._vendor_master is None:
            return None
        master = self._vendor_master
        assessment, elapsed = self._timed(
            lambda: assess_vendor_risk(invoice, master.known_vendors())
        )
        self._record(
            trace,
            "vendor-risk",
            f"{assessment.kind.value} ({assessment.severity.value})",
            elapsed,
        )
        return assessment
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/application/test_review_payable.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/application/review_payable.py tests/unit/application/test_review_payable.py
git commit -m "feat(review): vendor-risk step holds on HIGH BEC severity"
```

---

### Task 6: Settings + bootstrap wiring

**Files:**
- Modify: `src/apverify/infrastructure/settings.py`
- Modify: `src/apverify/interface/cli/bootstrap.py`
- Test: `tests/unit/interface/test_bootstrap.py` (append; create if absent)

**Interfaces:**
- Consumes: `apverify.infrastructure.vendor_master.repository.load_vendor_master`, `Settings.vendor_master_path`
- Produces: `build_review_use_case(..., vendor_master: VendorMasterRepository | None = None)` wires the master from `settings.vendor_master_path` when set.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/interface/test_bootstrap.py`:

```python
from apverify.interface.cli.bootstrap import _build_vendor_master
from apverify.infrastructure.settings import Settings


def test_vendor_master_is_none_when_unset(monkeypatch) -> None:
    monkeypatch.setenv("VENDOR_MASTER_PATH", "")
    # Settings requires GEMINI_API_KEY; provide a dummy so construction succeeds.
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert _build_vendor_master(Settings()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/interface/test_bootstrap.py -k vendor_master -v`
Expected: FAIL with `ImportError: cannot import name '_build_vendor_master'`

- [ ] **Step 3: Implement**

In `src/apverify/infrastructure/settings.py`, add a field with the others:

```python
    vendor_master_path: str = Field("", alias="VENDOR_MASTER_PATH")
```

In `src/apverify/interface/cli/bootstrap.py`, add imports:

```python
from pathlib import Path

from apverify.application.ports import VendorMasterRepository
from apverify.infrastructure.vendor_master.repository import load_vendor_master
```

Add the param to `build_review_use_case` and pass it through:

```python
def build_review_use_case(
    procurement: ProcurementRepository | None = None,
    settings: Settings | None = None,
    enable_audit: bool = False,
    enable_cross_check: bool = False,
    vendor_master: VendorMasterRepository | None = None,
) -> ReviewPayableUseCase:
    settings = settings or Settings()
    return ReviewPayableUseCase(
        renderer=Pdf2ImageRenderer(),
        extractor=_build_extractor(settings),
        ocr=TesseractOcrProvider(),
        procurement=procurement or InMemoryProcurementRepository(),
        auditor=_build_auditor(settings) if enable_audit else None,
        secondary_extractor=_build_secondary_extractor(settings) if enable_cross_check else None,
        vendor_master=vendor_master or _build_vendor_master(settings),
        tracer=_build_tracer(settings),
    )


def _build_vendor_master(settings: Settings) -> VendorMasterRepository | None:
    if not settings.vendor_master_path:
        return None
    return load_vendor_master(Path(settings.vendor_master_path))
```

(If `Path` is already imported in bootstrap, do not duplicate the import.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/interface/test_bootstrap.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/infrastructure/settings.py src/apverify/interface/cli/bootstrap.py tests/unit/interface/test_bootstrap.py
git commit -m "feat(bootstrap): wire vendor master from VENDOR_MASTER_PATH"
```

---

### Task 7: BEC synthesis — `eval/bec_synthesis.py`

**Files:**
- Create: `src/apverify/eval/bec_synthesis.py`
- Test: `tests/unit/eval/test_bec_synthesis.py`

**Interfaces:**
- Consumes: `apverify.eval.synthetic.{GroundTruth, generate_dataset}`, `apverify.domain.invoice.Invoice`, `apverify.domain.vendor_master.KnownVendor`
- Produces:
  - `BecCase(invoice: Invoice, master: tuple[KnownVendor, ...], scenario: str)` (frozen)
  - `build_bec_cases(base: Sequence[GroundTruth]) -> list[BecCase]`
  - scenario constants `BANK_CHANGE`, `IMPERSONATION`, `NEW_PAYEE`, `KNOWN_CLEAN`, `LEGIT_NEW` and `HIGH_SCENARIOS = (BANK_CHANGE, IMPERSONATION)`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_bec_synthesis.py`:

```python
from __future__ import annotations

from apverify.eval.bec_synthesis import SCENARIOS, build_bec_cases
from apverify.eval.synthetic import generate_dataset


def test_each_base_invoice_yields_one_case_per_scenario() -> None:
    cases = build_bec_cases(generate_dataset(count=3))
    scenarios = {case.scenario for case in cases}
    assert scenarios == set(SCENARIOS)
    assert all(case.master for case in cases)  # a master is always supplied


def test_known_clean_uses_a_known_bank_account() -> None:
    case = next(c for c in build_bec_cases(generate_dataset(count=1)) if c.scenario == "known_clean")
    known = {acct for vendor in case.master for acct in vendor.bank_accounts}
    assert case.invoice.bank_account in known


def test_impersonation_name_differs_from_every_known_vendor() -> None:
    case = next(
        c for c in build_bec_cases(generate_dataset(count=1)) if c.scenario == "impersonation"
    )
    assert all(case.invoice.vendor_name != vendor.name for vendor in case.master)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_bec_synthesis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.eval.bec_synthesis'`

- [ ] **Step 3: Implement `eval/bec_synthesis.py`**

```python
"""Synthesise a labelled BEC benchmark over base invoices.

We build a vendor master (each base vendor with one known bank account), then for each
base invoice emit the attack variants — a changed bank account, a typo-squatted vendor
name, a brand-new payee — and the legitimate look-alikes that must not raise a HIGH
flag: the same vendor paid to its known account, and a genuinely unrelated new vendor.

Deterministic: fixed transforms of the base, no randomness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from apverify.domain.invoice import Invoice
from apverify.domain.vendor_master import KnownVendor
from apverify.eval.synthetic import GroundTruth

BANK_CHANGE = "bank_change"
IMPERSONATION = "impersonation"
NEW_PAYEE = "new_payee"
KNOWN_CLEAN = "known_clean"
LEGIT_NEW = "legit_new"

SCENARIOS = (BANK_CHANGE, IMPERSONATION, NEW_PAYEE, KNOWN_CLEAN, LEGIT_NEW)
HIGH_SCENARIOS = (BANK_CHANGE, IMPERSONATION)

_ATTACKER_BANK = "ACCT-9999"
_NEW_BANK = "ACCT-7777"
# Letter -> confusable digit for typo-squatting a known name.
_LOOKALIKE = (("e", "3"), ("o", "0"), ("a", "4"), ("i", "1"))


@dataclass(frozen=True, slots=True)
class BecCase:
    invoice: Invoice
    master: tuple[KnownVendor, ...]
    scenario: str


def build_bec_cases(base: Sequence[GroundTruth]) -> list[BecCase]:
    names = sorted({truth.invoice.vendor_name for truth in base})
    account_of = {name: f"ACCT-{index:04d}" for index, name in enumerate(names)}
    master = tuple(
        KnownVendor(name, frozenset({account_of[name]})) for name in names
    )

    cases: list[BecCase] = []
    for index, truth in enumerate(base):
        invoice = truth.invoice
        known_account = account_of[invoice.vendor_name]
        cases.extend(
            [
                BecCase(replace(invoice, bank_account=_ATTACKER_BANK), master, BANK_CHANGE),
                BecCase(
                    replace(
                        invoice,
                        vendor_name=_typosquat(invoice.vendor_name),
                        bank_account=_ATTACKER_BANK,
                    ),
                    master,
                    IMPERSONATION,
                ),
                BecCase(
                    replace(
                        invoice,
                        vendor_name=f"New Supplier {index}",
                        bank_account=_NEW_BANK,
                    ),
                    master,
                    NEW_PAYEE,
                ),
                BecCase(replace(invoice, bank_account=known_account), master, KNOWN_CLEAN),
                BecCase(
                    replace(
                        invoice,
                        vendor_name=f"Unrelated Trader {index}",
                        bank_account=_NEW_BANK,
                    ),
                    master,
                    LEGIT_NEW,
                ),
            ]
        )
    return cases


def _typosquat(name: str) -> str:
    """One confusable substitution — looks like the vendor, is not the vendor."""
    lowered = name.lower()
    for original, lookalike in _LOOKALIKE:
        index = lowered.find(original)
        if index != -1:
            return name[:index] + lookalike + name[index + 1 :]
    return name + "X"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/eval/test_bec_synthesis.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/bec_synthesis.py tests/unit/eval/test_bec_synthesis.py
git commit -m "feat(eval): synthetic BEC injection with hard negatives"
```

---

### Task 8: BEC metrics — `eval/bec_eval.py`

**Files:**
- Create: `src/apverify/eval/bec_eval.py`
- Test: `tests/unit/eval/test_bec_eval.py`

**Interfaces:**
- Consumes: `apverify.domain.vendor_master.{assess_vendor_risk, Severity}`, `apverify.eval.bec_synthesis.{BecCase, HIGH_SCENARIOS, IMPERSONATION, LEGIT_NEW}`, `apverify.eval.fusion.auroc`
- Produces:
  - `BecReport(case_count: int, catch_rate: float, false_positive_rate: float, precision: float, impersonation_auroc: float, per_kind: dict[str, float], threshold: float)` (frozen)
  - `evaluate_bec(cases: Sequence[BecCase], threshold: float = 0.85) -> BecReport`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_bec_eval.py`:

```python
from __future__ import annotations

from apverify.eval.bec_eval import evaluate_bec
from apverify.eval.bec_synthesis import build_bec_cases
from apverify.eval.synthetic import generate_dataset


def test_bank_change_and_impersonation_are_caught_with_no_false_positives() -> None:
    report = evaluate_bec(build_bec_cases(generate_dataset(count=5)))
    assert report.per_kind["bank_change"] == 1.0
    assert report.per_kind["impersonation"] == 1.0
    assert report.false_positive_rate == 0.0


def test_new_payee_is_never_flagged_high() -> None:
    report = evaluate_bec(build_bec_cases(generate_dataset(count=5)))
    assert report.per_kind["new_payee"] == 0.0
    assert report.per_kind["legit_new"] == 0.0


def test_impersonation_score_separates_from_legitimate_new_vendors() -> None:
    report = evaluate_bec(build_bec_cases(generate_dataset(count=5)))
    assert report.impersonation_auroc >= 0.9


def test_empty_cases_yield_a_zeroed_report() -> None:
    report = evaluate_bec([])
    assert report.case_count == 0
    assert report.catch_rate == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_bec_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.eval.bec_eval'`

- [ ] **Step 3: Implement `eval/bec_eval.py`**

```python
"""Score the vendor-master assessor against the labelled BEC benchmark.

The HIGH-flag rate per scenario is the headline: bank-change and impersonation should
flag HIGH every time, while the legitimate scenarios (a known vendor paid to its known
account, a genuinely new vendor) must never flag HIGH. AUROC on the name-similarity
score over impersonation vs legitimate-new measures the one continuous boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.domain.vendor_master import Severity, assess_vendor_risk
from apverify.eval.bec_synthesis import HIGH_SCENARIOS, IMPERSONATION, LEGIT_NEW, BecCase
from apverify.eval.fusion import auroc


@dataclass(frozen=True, slots=True)
class BecReport:
    case_count: int
    catch_rate: float
    false_positive_rate: float
    precision: float
    impersonation_auroc: float
    per_kind: dict[str, float]
    threshold: float


def evaluate_bec(cases: Sequence[BecCase], threshold: float = 0.85) -> BecReport:
    results = [
        (assess_vendor_risk(case.invoice, case.master, threshold), case.scenario)
        for case in cases
    ]
    flagged = [(assessment.severity is Severity.HIGH, scenario) for assessment, scenario in results]

    high_cases = [f for f, scenario in flagged if scenario in HIGH_SCENARIOS]
    legit_cases = [f for f, scenario in flagged if scenario not in HIGH_SCENARIOS]
    flagged_high = [(f, scenario) for f, scenario in flagged if f]

    impersonation_samples = [
        (assessment.score, scenario == IMPERSONATION)
        for assessment, scenario in results
        if scenario in (IMPERSONATION, LEGIT_NEW)
    ]

    return BecReport(
        case_count=len(cases),
        catch_rate=_rate([f for f in high_cases if f], high_cases),
        false_positive_rate=_rate([f for f in legit_cases if f], legit_cases),
        precision=_rate(
            [f for f, scenario in flagged_high if scenario in HIGH_SCENARIOS], flagged_high
        ),
        impersonation_auroc=auroc(impersonation_samples),
        per_kind=_per_kind(flagged),
        threshold=threshold,
    )


def _rate(hits: Sequence[object], population: Sequence[object]) -> float:
    return len(hits) / len(population) if population else 0.0


def _per_kind(flagged: Sequence[tuple[bool, str]]) -> dict[str, float]:
    scenarios = sorted({scenario for _, scenario in flagged})
    return {
        scenario: _rate(
            [f for f, s in flagged if s == scenario and f],
            [f for f, s in flagged if s == scenario],
        )
        for scenario in scenarios
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/eval/test_bec_eval.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/bec_eval.py tests/unit/eval/test_bec_eval.py
git commit -m "feat(eval): BEC benchmark metrics"
```

---

### Task 9: `render_bec` in `eval/report.py`

**Files:**
- Modify: `src/apverify/eval/report.py`
- Test: `tests/unit/eval/test_report.py` (append)

**Interfaces:**
- Consumes: `apverify.eval.bec_eval.BecReport`
- Produces: `render_bec(report: BecReport, console: Console | None = None) -> None`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/eval/test_report.py`:

```python
from apverify.eval.bec_eval import BecReport
from apverify.eval.report import render_bec


def test_render_bec_prints_catch_and_false_positive() -> None:
    report = BecReport(
        case_count=25,
        catch_rate=1.0,
        false_positive_rate=0.0,
        precision=1.0,
        impersonation_auroc=0.98,
        per_kind={"bank_change": 1.0, "known_clean": 0.0},
        threshold=0.85,
    )
    console = Console(record=True, width=100)
    render_bec(report, console)
    text = console.export_text()
    assert "bank_change" in text
    assert "BEC" in text or "vendor" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/eval/test_report.py -k render_bec -v`
Expected: FAIL with `ImportError: cannot import name 'render_bec'`

- [ ] **Step 3: Implement**

Add the import in `src/apverify/eval/report.py` (with the other eval imports):

```python
from apverify.eval.bec_eval import BecReport
```

Add the function (place near `render_fraud`):

```python
def render_bec(report: BecReport, console: Console | None = None) -> None:
    console = console or Console()
    if report.case_count == 0:
        console.print("[yellow]No BEC cases to evaluate.[/yellow]")
        return

    console.print(
        f"[bold]Vendor-master / BEC detection[/bold] (n={report.case_count}): at name "
        f"threshold {report.threshold:.2f}, [green]{report.catch_rate:.0%}[/green] of "
        f"bank-change + impersonation caught at "
        f"[green]{report.false_positive_rate:.0%}[/green] false-positive "
        f"(precision {report.precision:.0%}, impersonation AUROC "
        f"{report.impersonation_auroc:.3f})."
    )

    table = Table(title="HIGH-flag rate by scenario", title_justify="left")
    table.add_column("Scenario", style="cyan")
    table.add_column("Flagged HIGH", justify="right")
    for scenario, rate in report.per_kind.items():
        table.add_row(scenario, f"{rate:.0%}")
    console.print(table)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/eval/test_report.py -k render_bec -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/report.py tests/unit/eval/test_report.py
git commit -m "feat(report): render BEC benchmark"
```

---

### Task 10: `apverify-eval-bec` CLI + console script

**Files:**
- Create: `src/apverify/eval/bec_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/eval/test_bec_cli.py`

**Interfaces:**
- Consumes: `evaluate_bec`, `build_bec_cases`, `render_bec`, `generate_dataset`
- Produces: `apverify-eval-bec` Typer app with `--count`, `--threshold`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/eval/test_bec_cli.py`:

```python
from __future__ import annotations

from typer.testing import CliRunner

from apverify.eval.bec_cli import app


def test_cli_runs_the_bec_benchmark() -> None:
    result = CliRunner().invoke(app, ["--count", "5"])
    assert result.exit_code == 0
    assert "BEC" in result.stdout or "vendor" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/eval/test_bec_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.eval.bec_cli'`

- [ ] **Step 3: Implement `eval/bec_cli.py`**

```python
"""``apverify-eval-bec`` — vendor-master / bank-change / impersonation benchmark.

Synthetic only: DocILE ground truth carries no bank-account data, so a BEC benchmark
cannot be built from it.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.bec_eval import evaluate_bec
from apverify.eval.bec_synthesis import build_bec_cases
from apverify.eval.report import render_bec
from apverify.eval.synthetic import generate_dataset

app = typer.Typer(add_completion=False, help="Vendor-master / BEC detection benchmark.")


@app.command()
def run(
    count: Annotated[int, typer.Option(help="Synthetic base invoices.")] = 25,
    threshold: Annotated[float, typer.Option(help="Impersonation name-match threshold.")] = 0.85,
) -> None:
    report = evaluate_bec(build_bec_cases(generate_dataset(count=count)), threshold=threshold)
    render_bec(report, Console())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/eval/test_bec_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Register the console script**

In `pyproject.toml`, under `[project.scripts]`, add:

```toml
apverify-eval-bec = "apverify.eval.bec_cli:app"
```

Reinstall: `pip install -e .`

- [ ] **Step 6: Gate + smoke-run + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`
Then: `apverify-eval-bec --count 25`
Expected: a BEC table with `bank_change 100%`, `impersonation 100%`, `known_clean 0%`, `new_payee 0%`, false-positive 0%.

```bash
git add src/apverify/eval/bec_cli.py tests/unit/eval/test_bec_cli.py pyproject.toml
git commit -m "feat(cli): apverify-eval-bec benchmark command"
```

---

## Final verification (after all tasks)

- [ ] `ruff check . && ruff format --check . && mypy --strict src tests` — clean
- [ ] `pytest -q` — all pass; domain layer 100% (`pytest --cov=apverify.domain --cov-report=term-missing`)
- [ ] `apverify-eval-bec --count 25` — bank-change + impersonation 100% HIGH, known_clean + new_payee 0% HIGH, false-positive 0%, impersonation AUROC ≥ 0.9
- [ ] Pipeline HOLD verified by Task 5's use-case test
- [ ] README: add a vendor-master / BEC paragraph + the CLI in the usage list (small follow-up, mirrors slice 1)

## Spec coverage check

- Eval + pipeline wiring → Tasks 3–6 (port, adapter, use-case step, bootstrap) ✓
- Hybrid check (kind+severity tier + name-similarity score) → Task 1 ✓
- Three signals severity-tiered (bank-change/impersonation HIGH, new-payee LOW) → Tasks 1, 2 ✓
- HIGH→HOLD, LOW→reason only → Task 2 ✓
- Canonical (non-folded) name match → Task 1 (Global Constraints) ✓
- Synthetic-only BEC benchmark + hard negatives → Task 7 ✓
- Per-signal catch/FP/precision + impersonation AUROC → Task 8 ✓
- XAI reasons per kind → Task 1 ✓
- Live `apverify review` integration + audit trace → Tasks 5, 6 ✓
- Acceptance (100%@0%-FP HIGH, new-payee never HIGH, AUROC ≥0.9, pipeline HOLD, domain 100%) → Tasks 5, 8 + Final verification ✓
