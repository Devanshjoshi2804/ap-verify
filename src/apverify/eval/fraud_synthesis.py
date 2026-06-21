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
        cases.extend(
            [
                _case(original, ledger, EXACT_RESEND),
                _case(_ocr_variant(original), ledger, OCR_VARIANT),
                _case(_small_edit(original), ledger, SMALL_EDIT),
                _case(original, ledger, MULTI_CHANNEL_RESEND),
                _case(_recurring(original, index), ledger, LEGIT_RECURRING),
                _case(_unrelated(original, index), ledger, LEGIT_DISTINCT),
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
    # A genuinely different invoice: new vendor, number and amount, so it matches
    # nothing in the ledger — the everyday case that must never be flagged.
    amount = Money(Decimal(1) + invoice.total.amount * 7)
    return replace(
        invoice,
        vendor_name=f"Unrelated Trader {index}",
        invoice_number=f"UNREL-{index}",
        subtotal=amount,
        total=amount,
    )
