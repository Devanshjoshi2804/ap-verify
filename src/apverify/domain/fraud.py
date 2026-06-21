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

from collections.abc import Sequence
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
    number_score = (
        1.0 if number_raw else 0.85 if number_fold else _number_similarity(candidate, other)
    )
    score = (
        _WEIGHTS["number"] * number_score
        + _WEIGHTS["amount"] * amount_proximity
        + _WEIGHTS["date"] * (1.0 if date_same else 0.0)
        + _WEIGHTS["vendor"] * vendor_sim
    )
    return DuplicateMatch(prior.identifier, tier, round(score, 4), _reason(tier, candidate, other))


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
    # The amounts differ, so at least one is non-zero and this denominator is positive.
    high = max(abs(a.amount), abs(b.amount))
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
