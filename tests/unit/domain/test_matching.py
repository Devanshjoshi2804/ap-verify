from __future__ import annotations

from tests.support import (
    PO_NUMBER,
    build_goods_receipt,
    build_invoice,
    build_purchase_order,
)

from apverify.domain.invoice import LineItem
from apverify.domain.matching import MatchOutcome, MatchStatus, three_way_match
from apverify.domain.procurement import GoodsReceiptLine, PurchaseOrderLine
from apverify.domain.value_objects import Gstin, Money


def test_clean_invoice_matches_its_po_and_grn() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    report = three_way_match(invoice, build_purchase_order(), build_goods_receipt())

    assert report.outcome is MatchOutcome.MATCHED
    assert report.mismatches == ()


def test_missing_purchase_order_is_reported() -> None:
    report = three_way_match(build_invoice(purchase_order_ref=PO_NUMBER), None)

    assert report.outcome is MatchOutcome.NO_PURCHASE_ORDER


def test_vendor_mismatch_is_caught() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    purchase_order = build_purchase_order(
        vendor_name="Totally Different Traders", vendor_gstin=None
    )
    report = three_way_match(invoice, purchase_order, build_goods_receipt())

    assert report.outcome is MatchOutcome.MISMATCH


def test_over_billing_quantity_is_caught() -> None:
    invoice = build_invoice(
        purchase_order_ref=PO_NUMBER,
        line_items=(
            LineItem("TMT Steel Bars (12mm)", 5, Money.of("78050.00"), Money.of("390250.00")),
        ),
    )
    report = three_way_match(invoice, build_purchase_order(), build_goods_receipt())

    assert report.outcome is MatchOutcome.MISMATCH
    assert any("exceeds ordered" in lm.detail for lm in report.line_matches)


def test_inflated_unit_price_is_caught() -> None:
    invoice = build_invoice(
        purchase_order_ref=PO_NUMBER,
        line_items=(
            LineItem("TMT Steel Bars (12mm)", 2, Money.of("90000.00"), Money.of("180000.00")),
        ),
        subtotal=Money.of("180000.00"),
    )
    report = three_way_match(invoice, build_purchase_order(), build_goods_receipt())

    assert report.outcome is MatchOutcome.MISMATCH


def test_partial_delivery_is_flagged_partial_not_mismatch() -> None:
    invoice = build_invoice(
        purchase_order_ref=PO_NUMBER,
        line_items=(
            LineItem("TMT Steel Bars (12mm)", 1, Money.of("78050.00"), Money.of("78050.00")),
        ),
        subtotal=Money.of("78050.00"),
    )
    grn = build_goods_receipt(lines=(GoodsReceiptLine("TMT Steel Bars (12mm)", 1),))
    report = three_way_match(invoice, build_purchase_order(), grn)

    assert report.outcome is MatchOutcome.PARTIAL


def test_wrong_po_reference_is_caught() -> None:
    invoice = build_invoice(purchase_order_ref="PO-9999")
    report = three_way_match(invoice, build_purchase_order(), build_goods_receipt())

    assert report.outcome is MatchOutcome.MISMATCH


def test_matches_with_a_po_but_no_goods_receipt() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    report = three_way_match(invoice, build_purchase_order(), None)

    assert report.outcome is MatchOutcome.MATCHED


def test_vendor_gstin_mismatch_is_caught() -> None:
    other_base = "29AAGCB1234M1Z"
    other_gstin = other_base + Gstin.compute_check_digit(other_base)
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    report = three_way_match(
        invoice, build_purchase_order(vendor_gstin=other_gstin), build_goods_receipt()
    )

    assert report.outcome is MatchOutcome.MISMATCH


def test_invoice_without_a_po_citation_reports_missing_reference() -> None:
    invoice = build_invoice(purchase_order_ref=None)
    report = three_way_match(invoice, build_purchase_order(), build_goods_receipt())

    assert any(
        finding.dimension.value == "po_reference" and finding.status is MatchStatus.MISSING
        for finding in report.findings
    )


def test_billing_more_than_was_received_is_caught() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)  # bills the full ordered qty of 2
    grn = build_goods_receipt(lines=(GoodsReceiptLine("TMT Steel Bars (12mm)", 1),))
    report = three_way_match(invoice, build_purchase_order(), grn)

    assert report.outcome is MatchOutcome.MISMATCH
    assert any("exceeds received" in lm.detail for lm in report.line_matches)


def test_zero_priced_line_matches_cleanly() -> None:
    invoice = build_invoice(
        purchase_order_ref=PO_NUMBER,
        line_items=(LineItem("Free sample", 1, Money.of(0), Money.of(0)),),
        subtotal=Money.of(0),
        tax=Money.of(0),
        total=Money.of(0),
    )
    purchase_order = build_purchase_order(
        subtotal=Money.of(0), lines=(PurchaseOrderLine("Free sample", 1, Money.of(0)),)
    )
    report = three_way_match(invoice, purchase_order, None)

    assert report.outcome is MatchOutcome.MATCHED


def test_line_not_on_po_is_caught() -> None:
    invoice = build_invoice(
        purchase_order_ref=PO_NUMBER,
        line_items=(LineItem("Mystery widget", 1, Money.of("156100.00"), Money.of("156100.00")),),
    )
    report = three_way_match(invoice, build_purchase_order(), build_goods_receipt())

    assert report.outcome is MatchOutcome.MISMATCH
    assert any(lm.status is MatchStatus.MISSING for lm in report.line_matches)
