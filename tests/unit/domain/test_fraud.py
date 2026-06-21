from __future__ import annotations

from decimal import Decimal

from apverify.domain.fraud import (
    DuplicateTier,
    IdentifiedInvoice,
    compare_invoices,
    find_duplicates,
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


def test_zero_total_invoices_compare_as_an_exact_resend() -> None:
    # Both amounts zero: proximity is defined as a perfect match, not a divide-by-zero.
    match = compare_invoices(_invoice(total="0.00"), _prior(_invoice(total="0.00")))
    assert match.tier is DuplicateTier.EXACT_RESEND
    assert match.score == 1.0


def test_unparseable_dates_fall_back_to_exact_string_equality() -> None:
    # When neither date parses, equal date strings still count as the same date.
    base = _invoice(date="n/a")
    match = compare_invoices(_invoice(date="n/a"), _prior(base))
    assert match.tier is DuplicateTier.EXACT_RESEND
