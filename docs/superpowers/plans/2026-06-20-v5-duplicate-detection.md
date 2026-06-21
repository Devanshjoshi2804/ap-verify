# Duplicate / near-duplicate detection (v5 slice 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure domain duplicate-invoice matcher with explainable tiers + a synthetic-fraud benchmark (catch-rate vs false-positive), eval-only.

**Architecture:** Pure `domain/fraud.py` matcher returns a discrete tier (the XAI reason) and a continuous score (the benchmark curve). An `eval/` harness injects duplicates + hard negatives over synthetic and DocILE-ground-truth invoices and reports catch-rate / false-positive / AUROC. No production-pipeline wiring this slice.

**Tech Stack:** Python 3.12, pydantic v2 entities (existing), `difflib` (stdlib) for fuzzy vendor match, `typer`/`rich` CLI, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-20-v5-duplicate-detection-design.md`

## Global Constraints

- Clean/hexagonal architecture: `domain` imports nothing from `application`/`infrastructure`/`eval`. `eval` may import `domain`. (verbatim dependency rule)
- No new third-party dependencies. `difflib`, `re`, `datetime`, `decimal` are stdlib and allowed.
- Determinism: no `random`, no wall-clock; synthetic values derive from the row index (matches `eval/synthetic.py`).
- Gates that must stay green after every task: `ruff check .`, `ruff format --check .`, `mypy --strict src tests`, `pytest`. **Domain layer must stay at 100% coverage.**
- No AI-tells; match surrounding code's comment density and idiom (engineering bar).
- **Git note:** the working dir is not currently a git repository, so the `git commit` step in each task is a no-op until `git init` is run. The real per-task checkpoint is the gate suite — run it where the commit step appears. Keep the commit commands in the plan for when version control is initialized.

---

### Task 1: Expose reusable normalisers from `domain/ocr.py`

The matcher needs two string normalisations already implemented privately as `_normalise`: a fold-free `canonical` (lower + strip non-alphanumerics) and the confusable-folded form. Expose both; keep existing behaviour identical.

**Files:**
- Modify: `src/apverify/domain/ocr.py`
- Test: `tests/unit/domain/test_ocr.py`

**Interfaces:**
- Produces: `canonical(value: str) -> str`, `fold_confusables(value: str) -> str`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/domain/test_ocr.py`:

```python
from apverify.domain.ocr import canonical, fold_confusables


def test_canonical_strips_punctuation_and_case_without_folding() -> None:
    assert canonical("INV-1001") == "inv1001"
    assert canonical("1,84,200") == "184200"


def test_fold_confusables_folds_ocr_lookalikes() -> None:
    # l→1, o→0, so an OCR misread of INV-1001 collapses onto the same key.
    assert fold_confusables("INV-l00l") == fold_confusables("INV-1001")
    assert fold_confusables("INV-1001") == "1nv1001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/domain/test_ocr.py -k "canonical or fold_confusables" -v`
Expected: FAIL with `ImportError: cannot import name 'canonical'`

- [ ] **Step 3: Implement the minimal change**

In `src/apverify/domain/ocr.py`, replace the private `_normalise` definition with two public functions and point existing callers at `fold_confusables`. Current code:

```python
def _normalise(value: str) -> str:
    return _NON_ALPHANUMERIC.sub("", value.lower()).translate(_CONFUSABLES)
```

Becomes:

```python
def canonical(value: str) -> str:
    """Lowercased alphanumerics only — formatting stripped, characters unchanged."""
    return _NON_ALPHANUMERIC.sub("", value.lower())


def fold_confusables(value: str) -> str:
    """``canonical`` with OCR-confusable characters folded to one form, so a value
    misread as ``O7AAECS`` still matches ``07AAECS``."""
    return canonical(value).translate(_CONFUSABLES)
```

Then replace the two internal uses of `_normalise(...)` (in `contains` and `contains_most_tokens`) with `fold_confusables(...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/domain/test_ocr.py -v`
Expected: PASS (new tests and all existing OCR tests)

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`
Expected: all green.

```bash
git add src/apverify/domain/ocr.py tests/unit/domain/test_ocr.py
git commit -m "refactor(ocr): expose canonical + fold_confusables for reuse"
```

---

### Task 2: Public `parse_date` in `domain/checks.py`

The matcher needs day-proximity, not just equality. `checks.py` already owns `_ACCEPTED_DATE_FORMATS` and a boolean `_parse_date`. Expose a `parse_date(value) -> date | None` and have the boolean delegate to it (DRY).

**Files:**
- Modify: `src/apverify/domain/checks.py`
- Test: `tests/unit/domain/test_checks.py`

**Interfaces:**
- Produces: `parse_date(value: str) -> datetime.date | None`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/domain/test_checks.py`:

```python
from datetime import date

from apverify.domain.checks import parse_date


def test_parse_date_reads_accepted_formats() -> None:
    assert parse_date("04-06-2025") == date(2025, 6, 4)
    assert parse_date("2025-06-04") == date(2025, 6, 4)


def test_parse_date_returns_none_for_unparseable() -> None:
    assert parse_date("not a date") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/domain/test_checks.py -k parse_date -v`
Expected: FAIL with `ImportError: cannot import name 'parse_date'`

- [ ] **Step 3: Implement**

In `src/apverify/domain/checks.py`, the current private helper is:

```python
def _parse_date(value: str) -> bool:
    for fmt in _ACCEPTED_DATE_FORMATS:
        try:
            datetime.strptime(value.strip(), fmt)
            return True
        except ValueError:
            continue
    return False
```

Replace it with a public parser the boolean delegates to:

```python
def parse_date(value: str) -> date | None:
    """The date in ``value`` under any accepted format, or ``None`` if none parse."""
    for fmt in _ACCEPTED_DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_date(value: str) -> bool:
    return parse_date(value) is not None
```

Add `date` to the existing `from datetime import ...` line (it currently imports `datetime`); make it `from datetime import date, datetime`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/domain/test_checks.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/domain/checks.py tests/unit/domain/test_checks.py
git commit -m "refactor(checks): expose parse_date returning a date"
```

---

### Task 3: Matcher types + `compare_invoices`

The heart: pairwise comparison → tier + score + reason.

**Files:**
- Create: `src/apverify/domain/fraud.py`
- Test: `tests/unit/domain/test_fraud.py`

**Interfaces:**
- Consumes: `apverify.domain.ocr.canonical`, `apverify.domain.ocr.fold_confusables`, `apverify.domain.checks.parse_date`, `apverify.domain.invoice.Invoice`, `apverify.domain.value_objects.Money`
- Produces:
  - `class DuplicateTier(Enum)`: `EXACT_RESEND`, `OCR_VARIANT`, `NEAR_DUPLICATE`, `DISTINCT`
  - `IdentifiedInvoice(identifier: str, invoice: Invoice)` (frozen)
  - `DuplicateMatch(matched_id: str, tier: DuplicateTier, score: float, reason: str)` (frozen)
  - `compare_invoices(candidate: Invoice, prior: IdentifiedInvoice, date_window_days: int = 3) -> DuplicateMatch`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/domain/test_fraud.py`:

```python
from __future__ import annotations

from decimal import Decimal

from apverify.domain.fraud import (
    DuplicateTier,
    IdentifiedInvoice,
    compare_invoices,
)
from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.value_objects import Money


def _invoice(
    *,
    vendor: str = "ACME Steel Pvt Ltd",
    number: str = "INV-2025-1001",
    date: str = "04-06-2025",
    total: str = "184200.00",
) -> Invoice:
    amount = Money(Decimal(total))
    return Invoice(
        vendor_name=vendor,
        invoice_number=number,
        invoice_date=date,
        currency="INR",
        subtotal=amount,
        tax=Money(Decimal("0")),
        total=amount,
        line_items=(LineItem("Widget", 1, amount, amount),),
        tax_breakdown=TaxBreakdown(),
        vendor_gstin="",
        purchase_order_ref="",
    )


def _prior(invoice: Invoice, identifier: str = "ledger-1") -> IdentifiedInvoice:
    return IdentifiedInvoice(identifier=identifier, invoice=invoice)


def test_identical_invoice_is_an_exact_resend_scoring_one() -> None:
    base = _invoice()
    match = compare_invoices(_invoice(), _prior(base))
    assert match.tier is DuplicateTier.EXACT_RESEND
    assert match.score == 1.0
    assert match.matched_id == "ledger-1"


def test_invoice_number_misread_is_an_ocr_variant() -> None:
    # INV-2025-1001 vs INV-2025-l00l: differs only by OCR-confusable characters.
    match = compare_invoices(_invoice(number="INV-2025-l00l"), _prior(_invoice()))
    assert match.tier is DuplicateTier.OCR_VARIANT
    assert "confusable" in match.reason.lower()


def test_edited_number_same_amount_and_date_is_a_near_duplicate() -> None:
    match = compare_invoices(_invoice(number="INV-2025-9999"), _prior(_invoice()))
    assert match.tier is DuplicateTier.NEAR_DUPLICATE


def test_slightly_edited_amount_same_everything_is_a_near_duplicate() -> None:
    match = compare_invoices(_invoice(total="184200.50"), _prior(_invoice()))
    assert match.tier is DuplicateTier.NEAR_DUPLICATE


def test_recurring_invoice_same_vendor_amount_new_date_is_distinct() -> None:
    # Monthly retainer: same vendor + amount, different invoice-no AND a later date.
    match = compare_invoices(
        _invoice(number="INV-2025-2002", date="04-07-2025"), _prior(_invoice())
    )
    assert match.tier is DuplicateTier.DISTINCT


def test_unrelated_invoice_is_distinct_with_a_low_score() -> None:
    other = _invoice(vendor="Konkan Foods Pvt Ltd", number="KF-77", total="500.00")
    match = compare_invoices(other, _prior(_invoice()))
    assert match.tier is DuplicateTier.DISTINCT
    assert match.score < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/domain/test_fraud.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.domain.fraud'`

- [ ] **Step 3: Implement `domain/fraud.py`**

```python
"""Duplicate / near-duplicate invoice detection — the first v5 fraud signal.

Duplicate fraud is the largest share of AP fraud: the same invoice resubmitted, OCR
noise across channels, or a small edit to slip past an exact-match check. This matcher
compares a candidate against a known prior and returns both a discrete *tier* (the
human-readable reason a flag ships with) and a continuous *score* (what the benchmark
sweeps into a catch-rate-vs-false-positive curve).

The hard case is telling a fraudulent near-duplicate from a legitimate recurring
charge (a monthly retainer: same vendor and amount, new invoice number and a later
date). The date is the discriminator — a resend shares the original date, a retainer
does not — so a differing date drops the pair to DISTINCT.

Pure domain logic: no ML, no I/O, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from enum import Enum

from apverify.domain.checks import parse_date
from apverify.domain.invoice import Invoice
from apverify.domain.ocr import canonical, fold_confusables
from apverify.domain.value_objects import Money

_VENDOR_MATCH = 0.9  # fuzzy-ratio floor for "same vendor"
_AMOUNT_NEAR = 0.98  # amount-proximity floor for a near-duplicate edit
_WEIGHTS = {"number": 0.4, "amount": 0.3, "date": 0.15, "vendor": 0.15}


class DuplicateTier(Enum):
    EXACT_RESEND = "exact_resend"
    OCR_VARIANT = "ocr_variant"
    NEAR_DUPLICATE = "near_duplicate"
    DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class IdentifiedInvoice:
    """A prior invoice plus the ledger id that distinguishes it from its (possibly
    shared) invoice number."""

    identifier: str
    invoice: Invoice


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    matched_id: str
    tier: DuplicateTier
    score: float
    reason: str


def compare_invoices(
    candidate: Invoice, prior: IdentifiedInvoice, date_window_days: int = 3
) -> DuplicateMatch:
    other = prior.invoice
    number_raw = canonical(candidate.invoice_number) == canonical(other.invoice_number)
    number_fold = fold_confusables(candidate.invoice_number) == fold_confusables(
        other.invoice_number
    )
    amount_proximity = _amount_proximity(candidate.total, other.total)
    amount_same = candidate.total == other.total
    vendor_sim = _vendor_similarity(candidate.vendor_name, other.vendor_name)
    vendor_same = vendor_sim >= _VENDOR_MATCH
    date_same = _dates_within(candidate.invoice_date, other.invoice_date, date_window_days)

    tier = _classify(
        number_raw=number_raw,
        number_fold=number_fold,
        amount_same=amount_same,
        amount_near=amount_proximity >= _AMOUNT_NEAR,
        vendor_same=vendor_same,
        date_same=date_same,
    )
    score = (
        _WEIGHTS["number"] * (1.0 if number_raw else 0.85 if number_fold else _number_similarity(candidate, other))
        + _WEIGHTS["amount"] * amount_proximity
        + _WEIGHTS["date"] * (1.0 if date_same else 0.0)
        + _WEIGHTS["vendor"] * vendor_sim
    )
    return DuplicateMatch(prior.identifier, tier, round(score, 4), _reason(tier, candidate, other))


def _classify(
    *,
    number_raw: bool,
    number_fold: bool,
    amount_same: bool,
    amount_near: bool,
    vendor_same: bool,
    date_same: bool,
) -> DuplicateTier:
    if not (vendor_same and date_same):
        return DuplicateTier.DISTINCT
    if amount_same and number_raw:
        return DuplicateTier.EXACT_RESEND
    if amount_same and number_fold:
        return DuplicateTier.OCR_VARIANT
    if amount_same or amount_near:
        return DuplicateTier.NEAR_DUPLICATE
    return DuplicateTier.DISTINCT


def _amount_proximity(a: Money, b: Money) -> float:
    if a.amount == b.amount:
        return 1.0
    high = max(abs(a.amount), abs(b.amount))
    if high == 0:
        return 1.0
    return float(Decimal(1) - min(Decimal(1), abs(a.amount - b.amount) / high))


def _vendor_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def _number_similarity(a: Invoice, b: Invoice) -> float:
    return SequenceMatcher(
        None, fold_confusables(a.invoice_number), fold_confusables(b.invoice_number)
    ).ratio()


def _dates_within(a: str, b: str, window_days: int) -> bool:
    parsed_a, parsed_b = parse_date(a), parse_date(b)
    if parsed_a is None or parsed_b is None:
        return canonical(a) == canonical(b)
    return abs((parsed_a - parsed_b).days) <= window_days


def _reason(tier: DuplicateTier, candidate: Invoice, other: Invoice) -> str:
    if tier is DuplicateTier.EXACT_RESEND:
        return (
            f"identical to prior invoice {other.invoice_number}: same vendor, "
            f"amount {other.total.amount}, date {other.invoice_date}"
        )
    if tier is DuplicateTier.OCR_VARIANT:
        return (
            f"same vendor + amount {other.total.amount} + date {other.invoice_date}; "
            f"invoice-no {candidate.invoice_number}<->{other.invoice_number} differs "
            f"only by OCR-confusable characters"
        )
    if tier is DuplicateTier.NEAR_DUPLICATE:
        return (
            f"near-duplicate of {other.invoice_number}: same vendor + date "
            f"{other.invoice_date}, amount/number edited "
            f"({candidate.total.amount} vs {other.total.amount})"
        )
    return f"distinct from {other.invoice_number}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/domain/test_fraud.py -v`
Expected: PASS (all six).

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/domain/fraud.py tests/unit/domain/test_fraud.py
git commit -m "feat(fraud): duplicate matcher with tiers, score and reasons"
```

---

### Task 4: `find_duplicates` over a ledger of priors

**Files:**
- Modify: `src/apverify/domain/fraud.py`
- Test: `tests/unit/domain/test_fraud.py`

**Interfaces:**
- Consumes: `compare_invoices`, `IdentifiedInvoice`, `DuplicateMatch`, `DuplicateTier`
- Produces: `find_duplicates(candidate: Invoice, priors: Sequence[IdentifiedInvoice], date_window_days: int = 3) -> list[DuplicateMatch]`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/domain/test_fraud.py`:

```python
from apverify.domain.fraud import find_duplicates


def test_find_duplicates_returns_non_distinct_matches_best_first() -> None:
    base = _invoice()
    priors = [
        _prior(_invoice(vendor="Konkan Foods Pvt Ltd", number="KF-1", total="9.0"), "unrelated"),
        _prior(_invoice(number="INV-2025-9999"), "near"),  # near-duplicate
        _prior(base, "exact"),  # exact resend, highest score
    ]
    matches = find_duplicates(_invoice(), priors)
    assert [m.matched_id for m in matches] == ["exact", "near"]  # unrelated dropped, sorted


def test_find_duplicates_with_no_priors_is_empty() -> None:
    assert find_duplicates(_invoice(), []) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/domain/test_fraud.py -k find_duplicates -v`
Expected: FAIL with `ImportError: cannot import name 'find_duplicates'`

- [ ] **Step 3: Implement**

Add to `src/apverify/domain/fraud.py` (add `from collections.abc import Sequence` to the imports):

```python
def find_duplicates(
    candidate: Invoice,
    priors: Sequence[IdentifiedInvoice],
    date_window_days: int = 3,
) -> list[DuplicateMatch]:
    """Every prior the candidate is not DISTINCT from, most-similar first."""
    matches = [
        match
        for prior in priors
        if (match := compare_invoices(candidate, prior, date_window_days)).tier
        is not DuplicateTier.DISTINCT
    ]
    return sorted(matches, key=lambda match: match.score, reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/domain/test_fraud.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/domain/fraud.py tests/unit/domain/test_fraud.py
git commit -m "feat(fraud): find_duplicates over a ledger of priors"
```

---

### Task 5: `InvoiceLedger` port (integration seam)

Defines the seam a future production wiring will implement; not consumed this slice.

**Files:**
- Modify: `src/apverify/application/ports.py`
- Test: `tests/unit/application/test_ports.py` (create if absent)

**Interfaces:**
- Consumes: `apverify.domain.fraud.IdentifiedInvoice`
- Produces: `InvoiceLedger` (runtime-checkable Protocol) with `known_invoices(self) -> tuple[IdentifiedInvoice, ...]`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/application/test_ports.py`:

```python
from apverify.application.ports import InvoiceLedger
from apverify.domain.fraud import IdentifiedInvoice


def test_a_simple_object_satisfies_the_invoice_ledger_port() -> None:
    class _Ledger:
        def known_invoices(self) -> tuple[IdentifiedInvoice, ...]:
            return ()

    assert isinstance(_Ledger(), InvoiceLedger)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/application/test_ports.py -k invoice_ledger -v`
Expected: FAIL with `ImportError: cannot import name 'InvoiceLedger'`

- [ ] **Step 3: Implement**

Add to `src/apverify/application/ports.py` (it already uses `@runtime_checkable`/`Protocol`; reuse those imports):

```python
from apverify.domain.fraud import IdentifiedInvoice


@runtime_checkable
class InvoiceLedger(Protocol):
    """Source of previously-seen invoices to check a candidate against for duplicates.

    Defined here as the integration seam; the duplicate benchmark supplies priors
    directly, and a production adapter (a persisted store of posted invoices) will
    implement this when the fraud stage is wired into the pipeline.
    """

    def known_invoices(self) -> tuple[IdentifiedInvoice, ...]: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/application/test_ports.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/application/ports.py tests/unit/application/test_ports.py
git commit -m "feat(ports): InvoiceLedger seam for duplicate detection"
```

---

### Task 6: Synthetic fraud injection — `eval/fraud_synthesis.py`

**Files:**
- Create: `src/apverify/eval/fraud_synthesis.py`
- Test: `tests/unit/eval/test_fraud_synthesis.py`

**Interfaces:**
- Consumes: `apverify.eval.synthetic.GroundTruth`, `apverify.eval.synthetic.generate_dataset`, `apverify.domain.invoice.Invoice`/`LineItem`/`TaxBreakdown`, `apverify.domain.value_objects.Money`, `apverify.domain.fraud.IdentifiedInvoice`
- Produces:
  - `FraudCase(candidate: Invoice, priors: tuple[IdentifiedInvoice, ...], is_fraud: bool, kind: str)` (frozen)
  - `build_fraud_cases(base: Sequence[GroundTruth]) -> list[FraudCase]`
  - kind constants: `EXACT_RESEND`, `OCR_VARIANT`, `SMALL_EDIT`, `MULTI_CHANNEL_RESEND`, `LEGIT_RECURRING`, `LEGIT_DISTINCT` (module-level `str`)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_fraud_synthesis.py`:

```python
from __future__ import annotations

from apverify.eval.fraud_synthesis import (
    FRAUD_KINDS,
    LEGIT_KINDS,
    build_fraud_cases,
)
from apverify.eval.synthetic import generate_dataset


def test_each_base_invoice_yields_one_case_per_kind() -> None:
    base = generate_dataset(count=4)
    cases = build_fraud_cases(base)
    kinds = {case.kind for case in cases}
    assert kinds == set(FRAUD_KINDS) | set(LEGIT_KINDS)
    assert all(case.priors for case in cases)  # every case has a ledger to check against


def test_fraud_kinds_are_labelled_fraud_and_legit_kinds_are_not() -> None:
    cases = build_fraud_cases(generate_dataset(count=4))
    for case in cases:
        assert case.is_fraud == (case.kind in FRAUD_KINDS)


def test_legit_recurring_keeps_vendor_and_amount_but_changes_date_and_number() -> None:
    base = generate_dataset(count=1)
    recurring = next(
        c for c in build_fraud_cases(base) if c.kind == "legit_recurring"
    )
    original = base[0].invoice
    assert recurring.candidate.vendor_name == original.vendor_name
    assert recurring.candidate.total == original.total
    assert recurring.candidate.invoice_date != original.invoice_date
    assert recurring.candidate.invoice_number != original.invoice_number
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_fraud_synthesis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.eval.fraud_synthesis'`

- [ ] **Step 3: Implement `eval/fraud_synthesis.py`**

```python
"""Synthesise a labelled duplicate-fraud benchmark over base invoices.

Labelled fraud data is scarce, so we inject it: for each base invoice we emit the
fraud variants a duplicate attack produces (verbatim resend, OCR-noise variant, small
edit, multi-channel resend) and the legitimate look-alikes a naive matcher would
wrongly flag (a recurring retainer, an unrelated invoice). The legitimate cases are
what make the false-positive number mean something.

Deterministic: every variant is a fixed transform of the base, no randomness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from apverify.domain.fraud import IdentifiedInvoice
from apverify.domain.invoice import Invoice
from apverify.domain.value_objects import Money
from apverify.eval.synthetic import GroundTruth

EXACT_RESEND = "exact_resend"
OCR_VARIANT = "ocr_variant"
SMALL_EDIT = "small_edit"
MULTI_CHANNEL_RESEND = "multi_channel_resend"
LEGIT_RECURRING = "legit_recurring"
LEGIT_DISTINCT = "legit_distinct"

FRAUD_KINDS = (EXACT_RESEND, OCR_VARIANT, SMALL_EDIT, MULTI_CHANNEL_RESEND)
LEGIT_KINDS = (LEGIT_RECURRING, LEGIT_DISTINCT)

# Digit -> OCR look-alike letter, the inverse of the critic's confusable folding.
_OCR_SWAP = str.maketrans({"0": "O", "1": "l"})


@dataclass(frozen=True, slots=True)
class FraudCase:
    candidate: Invoice
    priors: tuple[IdentifiedInvoice, ...]
    is_fraud: bool
    kind: str


def build_fraud_cases(base: Sequence[GroundTruth]) -> list[FraudCase]:
    ledger = tuple(
        IdentifiedInvoice(identifier=truth.label, invoice=truth.invoice) for truth in base
    )
    cases: list[FraudCase] = []
    for index, truth in enumerate(base):
        original = truth.invoice
        # An unrelated prior for the legit-distinct candidate: a different base invoice.
        unrelated = base[(index + 1) % len(base)].invoice
        cases.extend(
            [
                _case(original, ledger, EXACT_RESEND),
                _case(_ocr_variant(original), ledger, OCR_VARIANT),
                _case(_small_edit(original), ledger, SMALL_EDIT),
                _case(original, ledger, MULTI_CHANNEL_RESEND),
                _case(_recurring(original, index), ledger, LEGIT_RECURRING),
                _case(_unrelated(unrelated, index), ledger, LEGIT_DISTINCT),
            ]
        )
    return cases


def _case(candidate: Invoice, ledger: tuple[IdentifiedInvoice, ...], kind: str) -> FraudCase:
    return FraudCase(candidate, ledger, kind in FRAUD_KINDS, kind)


def _ocr_variant(invoice: Invoice) -> Invoice:
    return replace(invoice, invoice_number=invoice.invoice_number.translate(_OCR_SWAP))


def _small_edit(invoice: Invoice) -> Invoice:
    nudged = Money(invoice.total.amount + Decimal("0.50"))
    return replace(invoice, total=nudged)


def _recurring(invoice: Invoice, index: int) -> Invoice:
    # Next month's retainer: same vendor + amount, new number and a later date.
    return replace(
        invoice,
        invoice_number=f"{invoice.invoice_number}-R{index}",
        invoice_date="04-07-2025",
    )


def _unrelated(invoice: Invoice, index: int) -> Invoice:
    return replace(invoice, invoice_number=f"{invoice.invoice_number}-X{index}")
```

Note: `_small_edit` keeps the same date and number, so it stays a NEAR_DUPLICATE (amount within proximity); `_recurring` changes the date, dropping it to DISTINCT — the intended discriminator.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/eval/test_fraud_synthesis.py -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/fraud_synthesis.py tests/unit/eval/test_fraud_synthesis.py
git commit -m "feat(eval): synthetic duplicate-fraud injection with hard negatives"
```

---

### Task 7: Benchmark metrics — `eval/fraud_eval.py`

**Files:**
- Create: `src/apverify/eval/fraud_eval.py`
- Test: `tests/unit/eval/test_fraud_eval.py`

**Interfaces:**
- Consumes: `apverify.domain.fraud.find_duplicates`, `apverify.eval.fraud_synthesis.FraudCase`, `apverify.eval.fusion.auroc`, `apverify.eval.accuracy_eval.LabelledDocument`, `apverify.domain.invoice.Invoice`/`LineItem`/`TaxBreakdown`, `apverify.domain.value_objects.Money`
- Produces:
  - `FraudOperatingPoint(threshold: float, catch_rate: float, false_positive_rate: float)` (frozen)
  - `FraudReport(case_count: int, fraud_count: int, threshold: float, catch_rate: float, false_positive_rate: float, precision: float, auroc: float, sweep: tuple[FraudOperatingPoint, ...], per_kind: dict[str, float])` (frozen)
  - `evaluate_fraud(cases: Sequence[FraudCase], threshold: float | None = None) -> FraudReport`
  - `invoice_from_labelled(document: LabelledDocument) -> Invoice`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/eval/test_fraud_eval.py`:

```python
from __future__ import annotations

from apverify.eval.fraud_eval import evaluate_fraud
from apverify.eval.fraud_synthesis import build_fraud_cases
from apverify.eval.synthetic import generate_dataset


def test_exact_and_ocr_variants_are_caught_with_no_false_positives() -> None:
    report = evaluate_fraud(build_fraud_cases(generate_dataset(count=6)))
    # The unambiguous duplicate types are fully caught and no legit case is flagged.
    assert report.per_kind["exact_resend"] == 1.0
    assert report.per_kind["ocr_variant"] == 1.0
    assert report.false_positive_rate == 0.0


def test_recurring_retainer_is_never_flagged() -> None:
    report = evaluate_fraud(build_fraud_cases(generate_dataset(count=6)))
    assert report.per_kind["legit_recurring"] == 0.0


def test_score_separates_fraud_from_legitimate() -> None:
    report = evaluate_fraud(build_fraud_cases(generate_dataset(count=6)))
    assert report.auroc >= 0.9


def test_empty_cases_yield_a_zeroed_report() -> None:
    report = evaluate_fraud([])
    assert report.case_count == 0
    assert report.catch_rate == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/eval/test_fraud_eval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.eval.fraud_eval'`

- [ ] **Step 3: Implement `eval/fraud_eval.py`**

```python
"""Score the duplicate matcher against a labelled fraud benchmark.

For each case we take the candidate's best duplicate score against the ledger (0 if
the matcher finds nothing non-DISTINCT), then report the two numbers that matter for a
fraud control: catch-rate (recall on true duplicates) and false-positive-rate (legit
invoices wrongly flagged). A threshold sweep gives the catch-rate-vs-false-positive
curve and the safe (zero-false-positive) operating point, mirroring the v4 risk-
coverage view; AUROC summarises how well the score separates fraud from legitimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.domain.fraud import find_duplicates
from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.value_objects import Money
from apverify.eval.accuracy_eval import LabelledDocument
from apverify.eval.fraud_synthesis import FRAUD_KINDS, FraudCase
from apverify.eval.fusion import auroc


@dataclass(frozen=True, slots=True)
class FraudOperatingPoint:
    threshold: float
    catch_rate: float
    false_positive_rate: float


@dataclass(frozen=True, slots=True)
class FraudReport:
    case_count: int
    fraud_count: int
    threshold: float
    catch_rate: float
    false_positive_rate: float
    precision: float
    auroc: float
    sweep: tuple[FraudOperatingPoint, ...]
    per_kind: dict[str, float]


def evaluate_fraud(cases: Sequence[FraudCase], threshold: float | None = None) -> FraudReport:
    scored = [(_best_score(case), case.is_fraud, case.kind) for case in cases]
    samples = [(score, is_fraud) for score, is_fraud, _ in scored]
    sweep = _sweep(samples)
    chosen = threshold if threshold is not None else _zero_fp_threshold(sweep)
    flagged = [(score >= chosen, is_fraud, kind) for score, is_fraud, kind in scored]

    frauds = [s for s in flagged if s[1]]
    legit = [s for s in flagged if not s[1]]
    caught = [s for s in frauds if s[0]]
    false_pos = [s for s in legit if s[0]]
    flagged_total = [s for s in flagged if s[0]]

    return FraudReport(
        case_count=len(cases),
        fraud_count=len(frauds),
        threshold=chosen,
        catch_rate=len(caught) / len(frauds) if frauds else 0.0,
        false_positive_rate=len(false_pos) / len(legit) if legit else 0.0,
        precision=len(caught) / len(flagged_total) if flagged_total else 0.0,
        auroc=auroc(samples),
        sweep=sweep,
        per_kind=_per_kind(flagged),
    )


def invoice_from_labelled(document: LabelledDocument) -> Invoice:
    """Build an Invoice from a dataset document's ground-truth fields, so the benchmark
    runs on real invoices without a model call (quota-free realism check)."""
    truth = document.truth
    total = Money.of(_amount(truth.get("total", "0")))
    subtotal = Money.of(_amount(truth.get("subtotal", truth.get("total", "0"))))
    return Invoice(
        vendor_name=truth.get("vendor_name", ""),
        invoice_number=truth.get("invoice_number", ""),
        invoice_date=truth.get("invoice_date", ""),
        currency=truth.get("currency", "INR"),
        subtotal=subtotal,
        tax=Money.of(0),
        total=total,
        line_items=(LineItem("", 1, total, total),),
        tax_breakdown=TaxBreakdown(),
        vendor_gstin=truth.get("vendor_gstin", ""),
        purchase_order_ref="",
    )


def _best_score(case: FraudCase) -> float:
    matches = find_duplicates(case.candidate, case.priors)
    return matches[0].score if matches else 0.0


def _sweep(samples: Sequence[tuple[float, bool]], steps: int = 20) -> tuple[FraudOperatingPoint, ...]:
    frauds = [s for s in samples if s[1]]
    legit = [s for s in samples if not s[1]]
    points: list[FraudOperatingPoint] = []
    for step in range(steps + 1):
        threshold = step / steps
        caught = sum(1 for score, _ in frauds if score >= threshold)
        false_pos = sum(1 for score, _ in legit if score >= threshold)
        points.append(
            FraudOperatingPoint(
                threshold=threshold,
                catch_rate=caught / len(frauds) if frauds else 0.0,
                false_positive_rate=false_pos / len(legit) if legit else 0.0,
            )
        )
    return tuple(points)


def _zero_fp_threshold(sweep: Sequence[FraudOperatingPoint]) -> float:
    """The lowest threshold with no false positives (most catch at zero FP); 1.0 if
    none is clean."""
    clean = [point for point in sweep if point.false_positive_rate == 0.0]
    if not clean:
        return 1.0
    return min(clean, key=lambda point: point.threshold).threshold


def _per_kind(flagged: Sequence[tuple[bool, bool, str]]) -> dict[str, float]:
    kinds = sorted({kind for _, _, kind in flagged})
    result: dict[str, float] = {}
    for kind in kinds:
        rows = [is_flagged for is_flagged, _, k in flagged if k == kind]
        result[kind] = sum(1 for flag in rows if flag) / len(rows) if rows else 0.0
    return result


def _amount(value: str) -> float:
    try:
        return float(str(value).replace(",", "") or 0)
    except ValueError:
        return 0.0
```

Note: `_zero_fp_threshold` returns the lowest clean threshold. Because well-formed legit cases score 0, the default operating point falls just above 0 — catching every fraud whose score clears it at zero false positives. Confirm `LabelledDocument` exposes a `truth: dict[str, str]` attribute when implementing; if the attribute name differs, adapt `invoice_from_labelled` accordingly (it is only exercised by the DocILE CLI path, Task 9).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/eval/test_fraud_eval.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/fraud_eval.py tests/unit/eval/test_fraud_eval.py
git commit -m "feat(eval): duplicate-fraud benchmark metrics"
```

---

### Task 8: `render_fraud` in `eval/report.py`

**Files:**
- Modify: `src/apverify/eval/report.py`
- Test: `tests/unit/eval/test_report.py` (add a case; create if absent)

**Interfaces:**
- Consumes: `apverify.eval.fraud_eval.FraudReport`, `FraudOperatingPoint`
- Produces: `render_fraud(report: FraudReport, console: Console | None = None) -> None`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/eval/test_report.py`:

```python
from rich.console import Console

from apverify.eval.fraud_eval import FraudOperatingPoint, FraudReport
from apverify.eval.report import render_fraud


def test_render_fraud_prints_catch_and_false_positive() -> None:
    report = FraudReport(
        case_count=12,
        fraud_count=8,
        threshold=0.05,
        catch_rate=1.0,
        false_positive_rate=0.0,
        precision=1.0,
        auroc=0.97,
        sweep=(FraudOperatingPoint(0.05, 1.0, 0.0),),
        per_kind={"exact_resend": 1.0, "legit_recurring": 0.0},
    )
    console = Console(record=True, width=100)
    render_fraud(report, console)
    text = console.export_text()
    assert "catch" in text.lower()
    assert "exact_resend" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/eval/test_report.py -k render_fraud -v`
Expected: FAIL with `ImportError: cannot import name 'render_fraud'`

- [ ] **Step 3: Implement**

Add the import near the other eval imports in `src/apverify/eval/report.py`:

```python
from apverify.eval.fraud_eval import FraudReport
```

And add the function (place it near `render_fusion`):

```python
def render_fraud(report: FraudReport, console: Console | None = None) -> None:
    console = console or Console()
    if report.case_count == 0:
        console.print("[yellow]No fraud cases to evaluate.[/yellow]")
        return

    console.print(
        f"[bold]Duplicate-fraud detection[/bold] (n={report.case_count}, "
        f"{report.fraud_count} fraudulent): at score ≥{report.threshold:.2f}, "
        f"[green]{report.catch_rate:.0%}[/green] of duplicates caught at "
        f"[green]{report.false_positive_rate:.0%}[/green] false-positive "
        f"(precision {report.precision:.0%}, AUROC {report.auroc:.3f})."
    )

    by_kind = Table(title="Catch rate by kind", title_justify="left")
    by_kind.add_column("Kind", style="cyan")
    by_kind.add_column("Flagged", justify="right")
    for kind, rate in report.per_kind.items():
        by_kind.add_row(kind, f"{rate:.0%}")
    console.print(by_kind)

    curve = Table(title="Catch-rate vs false-positive sweep", title_justify="left")
    curve.add_column("Score ≥", justify="right")
    curve.add_column("Caught", justify="right")
    curve.add_column("False-pos", justify="right")
    for point in report.sweep:
        curve.add_row(
            f"{point.threshold:.2f}",
            f"{point.catch_rate:.0%}",
            f"{point.false_positive_rate:.0%}",
        )
    console.print(curve)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/eval/test_report.py -k render_fraud -v`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`

```bash
git add src/apverify/eval/report.py tests/unit/eval/test_report.py
git commit -m "feat(report): render duplicate-fraud benchmark"
```

---

### Task 9: `apverify-eval-fraud` CLI + console script

**Files:**
- Create: `src/apverify/eval/fraud_cli.py`
- Modify: `pyproject.toml` (add console script)
- Test: `tests/unit/eval/test_fraud_cli.py`

**Interfaces:**
- Consumes: `evaluate_fraud`, `build_fraud_cases`, `invoice_from_labelled`, `render_fraud`, `generate_dataset`, `load_docile_labelled`, `IdentifiedInvoice`, `FraudCase`
- Produces: `apverify-eval-fraud` Typer app with `--dataset synthetic|docile`, `--count`, `--dataset-path`, `--split`, `--limit`, `--threshold`

- [ ] **Step 1: Write the failing test (synthetic path, no I/O)**

Create `tests/unit/eval/test_fraud_cli.py`:

```python
from typer.testing import CliRunner

from apverify.eval.fraud_cli import app


def test_cli_runs_the_synthetic_benchmark() -> None:
    result = CliRunner().invoke(app, ["--dataset", "synthetic", "--count", "6"])
    assert result.exit_code == 0
    assert "Duplicate-fraud detection" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/eval/test_fraud_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'apverify.eval.fraud_cli'`

- [ ] **Step 3: Implement `eval/fraud_cli.py`**

```python
"""``apverify-eval-fraud`` — duplicate-fraud catch-rate vs false-positive benchmark.

Synthetic is the controlled headline (exact ground truth, crafted hard negatives);
DocILE is a realism check that builds invoices from ground-truth fields, so it needs
no model calls.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.domain.fraud import IdentifiedInvoice
from apverify.eval.accuracy_eval import load_docile_labelled
from apverify.eval.fraud_eval import evaluate_fraud, invoice_from_labelled
from apverify.eval.fraud_synthesis import FraudCase, build_fraud_cases
from apverify.eval.report import render_fraud
from apverify.eval.synthetic import generate_dataset

app = typer.Typer(add_completion=False, help="Duplicate-fraud detection benchmark.")


@app.command()
def run(
    dataset: Annotated[str, typer.Option(help="synthetic or docile.")] = "synthetic",
    count: Annotated[int, typer.Option(help="Synthetic base invoices.")] = 25,
    dataset_path: Annotated[str, typer.Option(help="DocILE path (required for docile).")] = "",
    split: Annotated[str, typer.Option(help="Dataset split.")] = "val",
    limit: Annotated[int, typer.Option(help="DocILE documents.")] = 50,
    threshold: Annotated[
        float, typer.Option(help="Flag threshold (default: zero-FP operating point).")
    ] = -1.0,
) -> None:
    console = Console()
    if dataset == "docile":
        if not dataset_path:
            raise typer.BadParameter("--dataset-path is required for docile")
        documents = load_docile_labelled(dataset_path, split=split, limit=limit)
        base = [invoice_from_labelled(document) for document in documents]
        cases = _cases_from_invoices(base)
    else:
        cases = build_fraud_cases(generate_dataset(count=count))

    report = evaluate_fraud(cases, threshold=None if threshold < 0 else threshold)
    render_fraud(report, console)


def _cases_from_invoices(invoices: list) -> list[FraudCase]:  # type: ignore[type-arg]
    # Wrap real invoices as a GroundTruth-like base so build_fraud_cases can inject.
    from apverify.eval.synthetic import GroundTruth

    base = [GroundTruth(label=f"docile-{i:03d}", invoice=inv) for i, inv in enumerate(invoices)]
    return build_fraud_cases(base)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/eval/test_fraud_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Register the console script**

In `pyproject.toml`, under `[project.scripts]` (where the other `apverify-eval-*` scripts are), add:

```toml
apverify-eval-fraud = "apverify.eval.fraud_cli:app"
```

Reinstall so the entry point resolves: `pip install -e .`

- [ ] **Step 6: Gate + smoke-run + commit**

Run: `ruff check . && ruff format --check . && mypy --strict src tests && pytest -q`
Then smoke-run: `apverify-eval-fraud --dataset synthetic --count 25`
Expected: a duplicate-fraud table with `exact_resend 100%`, `legit_recurring 0%`, false-positive 0%.

```bash
git add src/apverify/eval/fraud_cli.py tests/unit/eval/test_fraud_cli.py pyproject.toml
git commit -m "feat(cli): apverify-eval-fraud benchmark command"
```

---

## Final verification (after all tasks)

- [ ] `ruff check . && ruff format --check . && mypy --strict src tests` — all clean
- [ ] `pytest -q` — all pass, domain layer 100%
- [ ] `apverify-eval-fraud --dataset synthetic --count 25` — exact + OCR-variant caught 100% @ 0% false-positive; legit_recurring never flagged; score AUROC ≥ 0.9
- [ ] (optional, gated) `apverify-eval-fraud --dataset docile --dataset-path /tmp/docile-data --limit 50` — realism check reported honestly
- [ ] Update README with a duplicate-fraud section (catch-rate vs false-positive, the legit-recurring discrimination as the headline) — small follow-up, mirrors prior eval sections

## Spec coverage check

- Eval-only scope → Tasks 3–9 (no pipeline wiring); `InvoiceLedger` seam → Task 5 ✓
- Hybrid matcher (tier + score) → Task 3 ✓
- Tiers EXACT/OCR_VARIANT/NEAR_DUPLICATE/DISTINCT → Task 3 ✓
- Confusable-folded invoice-no, fuzzy vendor, amount proximity, date window → Tasks 1, 3 ✓
- Synthetic injection + hard negatives (legit_recurring/legit_distinct) → Task 6 ✓
- Metrics: catch-rate, FP-rate, precision, AUROC, sweep, per-kind → Task 7 ✓
- DocILE quota-free realism (invoice from ground-truth fields) → Tasks 7, 9 ✓
- XAI reasons per tier → Task 3 ✓
- Acceptance bar (100% exact/OCR @ 0% FP, legit-recurring 0 FP, AUROC ≥ 0.9, domain 100%) → Task 7 tests + Final verification ✓
