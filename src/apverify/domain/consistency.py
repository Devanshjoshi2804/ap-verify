"""Self-consistency: do two independent extractions of the same invoice agree?

If a second model — a different vendor, a different architecture — reads the page
and lands on the same vendor, GSTIN, and total, that agreement is strong evidence
the values are real. Where the two disagree, at least one is wrong, and the field
is no longer safe to trust. Pure comparison; the extractions are produced upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from enum import StrEnum

from apverify.domain.critique import InvoiceField
from apverify.domain.invoice import Invoice
from apverify.domain.value_objects import Money


class Agreement(StrEnum):
    AGREES = "agrees"
    DIFFERS = "differs"


@dataclass(frozen=True, slots=True)
class FieldComparison:
    field: InvoiceField
    agreement: Agreement
    primary: str
    secondary: str


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    comparisons: tuple[FieldComparison, ...]

    @property
    def disagreements(self) -> tuple[FieldComparison, ...]:
        return tuple(c for c in self.comparisons if c.agreement is Agreement.DIFFERS)


@dataclass(frozen=True, slots=True)
class ConsistencyPolicy:
    vendor_name_threshold: float = 0.85
    amount_tolerance: Decimal = Decimal("0.05")


DEFAULT_CONSISTENCY_POLICY = ConsistencyPolicy()


def compare_extractions(
    primary: Invoice, secondary: Invoice, policy: ConsistencyPolicy = DEFAULT_CONSISTENCY_POLICY
) -> ConsistencyReport:
    comparisons = (
        _text(InvoiceField.VENDOR, primary.vendor_name, secondary.vendor_name, fuzzy=policy),
        _text(InvoiceField.GSTIN, primary.vendor_gstin, secondary.vendor_gstin),
        _text(InvoiceField.INVOICE_NUMBER, primary.invoice_number, secondary.invoice_number),
        _text(InvoiceField.INVOICE_DATE, primary.invoice_date, secondary.invoice_date),
        _text(InvoiceField.CURRENCY, primary.currency, secondary.currency),
        _money(InvoiceField.SUBTOTAL, primary.subtotal, secondary.subtotal, policy),
        _money(InvoiceField.TAX, primary.tax, secondary.tax, policy),
        _money(InvoiceField.TOTAL, primary.total, secondary.total, policy),
    )
    return ConsistencyReport(comparisons=comparisons)


def _text(
    field: InvoiceField,
    primary: str | None,
    secondary: str | None,
    fuzzy: ConsistencyPolicy | None = None,
) -> FieldComparison:
    left, right = (primary or "").strip(), (secondary or "").strip()
    if fuzzy is not None:
        agrees = _similarity(left, right) >= fuzzy.vendor_name_threshold
    else:
        agrees = _canonical(left) == _canonical(right)
    return _comparison(field, agrees, left, right)


def _money(
    field: InvoiceField, primary: Money, secondary: Money, policy: ConsistencyPolicy
) -> FieldComparison:
    agrees = abs(primary.amount - secondary.amount) <= policy.amount_tolerance
    return _comparison(field, agrees, str(primary), str(secondary))


def _comparison(field: InvoiceField, agrees: bool, primary: str, secondary: str) -> FieldComparison:
    agreement = Agreement.AGREES if agrees else Agreement.DIFFERS
    return FieldComparison(field=field, agreement=agreement, primary=primary, secondary=secondary)


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _canonical(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())
