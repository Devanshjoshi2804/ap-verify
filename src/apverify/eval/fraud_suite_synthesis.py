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
_HISTORY_SPREAD = (
    Decimal("0.90"),
    Decimal("0.95"),
    Decimal("1.00"),
    Decimal("1.05"),
    Decimal("1.10"),
)


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
        # A fresh copy: new invoice-no + later date, so a same-vendor candidate is not a
        # duplicate of the ledger original (the date is the duplicate discriminator).
        fresh = replace(
            original, invoice_number=f"{original.invoice_number}-N", invoice_date=_NEW_DATE
        )
        candidates: list[tuple[Invoice, str]] = [
            (
                replace(
                    fresh,
                    bank_account=known_account,
                    total=_money(median * Decimal("1.02")),
                    subtotal=_money(median * Decimal("1.02")),
                ),
                CLEAN,
            ),
            (original, DUP_RESEND),
            (
                replace(original, invoice_number=original.invoice_number.translate(_OCR_SWAP)),
                DUP_OCR_VARIANT,
            ),
            (replace(fresh, bank_account=_ATTACKER_BANK), BANK_CHANGE),
            (
                replace(
                    fresh, vendor_name=_typosquat(original.vendor_name), bank_account=_ATTACKER_BANK
                ),
                IMPERSONATION,
            ),
            (_at(fresh, median * 10), AMOUNT_SPIKE),
            (_at(fresh, _just_under_round(median)), THRESHOLD_GAMING),
        ]
        cases.extend(
            SuiteCase(invoice, priors, master, history, label, label != CLEAN)
            for invoice, label in candidates
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
