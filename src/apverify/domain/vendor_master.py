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
    return _account(account) in {_account(known_account) for known_account in known}


def _account(value: str) -> str:
    return value.replace(" ", "").upper()


def _mask(account: str) -> str:
    digits = _account(account)
    return f"****{digits[-4:]}" if len(digits) >= 4 else digits
