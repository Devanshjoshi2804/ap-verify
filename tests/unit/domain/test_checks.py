from __future__ import annotations

from datetime import date

from tests.support import VALID_GSTIN, InvoiceFactory, RawTextFactory

from apverify.domain.checks import parse_date, review
from apverify.domain.critique import (
    ApprovalDecision,
    CheckCategory,
    CriticReport,
    InvoiceField,
)
from apverify.domain.value_objects import Money


def test_parse_date_reads_accepted_formats() -> None:
    assert parse_date("04-06-2025") == date(2025, 6, 4)
    assert parse_date("2025-06-04") == date(2025, 6, 4)


def test_parse_date_returns_none_for_unparseable() -> None:
    assert parse_date("not a date") is None


def test_zero_unit_price_on_a_nonzero_line_is_flagged() -> None:
    # A model returned junk (e.g. "Pkg.") where a unit price belongs; the mapping
    # coerces it to 0 to avoid crashing — the critic must not let that vanish silently.
    from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
    from apverify.domain.ocr import RawText

    amount = Money.of("100")
    invoice = Invoice(
        vendor_name="ACME Steel Pvt Ltd",
        invoice_number="INV-2025-0042",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=amount,
        tax=Money.of("0"),
        total=amount,
        line_items=(LineItem("Widget", 1, Money.of("0"), amount),),  # unit price 0, line total 100
        tax_breakdown=TaxBreakdown(),
        vendor_gstin=VALID_GSTIN,
    )
    # Page text reconciles the header amounts, so only the line-integrity check can fail.
    raw = RawText(
        text=(
            f"ACME Steel Pvt Ltd\nGSTIN {VALID_GSTIN}\nINV-2025-0042\n04-06-2025\n"
            "Widget\nSubtotal 100\nTotal INR 100.00"
        )
    )
    report = review(invoice, raw)

    assert report.decision is not ApprovalDecision.AUTO_APPROVE
    assert any("unit price" in flag.detail.lower() for flag in report.flags)


def _flag(report: CriticReport, category: CheckCategory, field: InvoiceField) -> bool:
    return any(flag.category is category and flag.field is field for flag in report.flags)


class TestCleanInvoice:
    def test_auto_approves_with_full_confidence(
        self, make_invoice: InvoiceFactory, make_raw_text: RawTextFactory
    ) -> None:
        invoice = make_invoice()
        report = review(invoice, make_raw_text(invoice))

        assert report.decision is ApprovalDecision.AUTO_APPROVE
        assert report.overall_confidence == 1.0
        assert report.flags == ()


class TestCorruptedTotal:
    def test_inconsistent_total_is_held_by_arithmetic(
        self, make_invoice: InvoiceFactory, make_raw_text: RawTextFactory
    ) -> None:
        # The page prints 1,84,000 (so OCR sees it) but the parts sum to 1,84,200.
        invoice = make_invoice(total=Money.of("184000.00"))
        report = review(invoice, make_raw_text(invoice))

        assert report.decision is ApprovalDecision.HOLD
        assert _flag(report, CheckCategory.ARITHMETIC, InvoiceField.TOTAL)

    def test_hallucinated_total_is_held_by_cross_check(
        self, make_invoice: InvoiceFactory, make_raw_text: RawTextFactory
    ) -> None:
        page = make_raw_text(make_invoice())  # page shows the correct total
        invoice = make_invoice(total=Money.of("999999.00"))  # model invented another
        report = review(invoice, page)

        assert report.decision is ApprovalDecision.HOLD
        assert _flag(report, CheckCategory.CROSS_CHECK, InvoiceField.TOTAL)


class TestFormatChecks:
    def test_invalid_gstin_checksum_is_flagged(
        self, make_invoice: InvoiceFactory, make_raw_text: RawTextFactory
    ) -> None:
        wrong_last = "A" if VALID_GSTIN[-1] != "A" else "B"
        invoice = make_invoice(vendor_gstin=VALID_GSTIN[:-1] + wrong_last)
        report = review(invoice, make_raw_text(invoice))

        assert report.decision is not ApprovalDecision.AUTO_APPROVE
        assert _flag(report, CheckCategory.FORMAT, InvoiceField.GSTIN)

    def test_unparseable_date_is_flagged(
        self, make_invoice: InvoiceFactory, make_raw_text: RawTextFactory
    ) -> None:
        invoice = make_invoice(invoice_date="31/31/2025")
        report = review(invoice, make_raw_text(invoice))

        assert _flag(report, CheckCategory.FORMAT, InvoiceField.INVOICE_DATE)

    def test_unknown_currency_is_flagged(
        self, make_invoice: InvoiceFactory, make_raw_text: RawTextFactory
    ) -> None:
        invoice = make_invoice(currency="XYZ")
        report = review(invoice, make_raw_text(invoice))

        assert _flag(report, CheckCategory.FORMAT, InvoiceField.CURRENCY)


class TestMissingOptionalField:
    def test_absent_gstin_yields_no_signal_and_does_not_block(
        self, make_invoice: InvoiceFactory, make_raw_text: RawTextFactory
    ) -> None:
        invoice = make_invoice(vendor_gstin=None)
        report = review(invoice, make_raw_text(invoice, omit=["vendor_gstin"]))

        reported_fields = {fc.field for fc in report.field_confidences}
        assert InvoiceField.GSTIN not in reported_fields
        assert report.decision is ApprovalDecision.AUTO_APPROVE
