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


def test_short_changed_bank_account_is_reported_unmasked() -> None:
    # A short account number has nothing to mask; the reason still names it.
    result = assess_vendor_risk(_invoice(bank="12"), _MASTER)
    assert result.kind is VendorRiskKind.BANK_CHANGE
    assert "12" in result.reason
