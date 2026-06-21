from __future__ import annotations

from tests.support import InvoiceFactory, RawTextFactory, build_invoice, build_raw_text

from apverify.domain.checks import review
from apverify.domain.critique import (
    DEFAULT_POLICY,
    ApprovalDecision,
    CheckCategory,
    InvoiceField,
    decide,
    overall_confidence,
)
from apverify.domain.invoice import TaxBreakdown
from apverify.domain.value_objects import Gstin, InvoiceNumber, Money


class TestArithmeticWithoutBreakdownOrLines:
    def test_skips_line_and_component_sums_when_absent(
        self, make_invoice: InvoiceFactory, make_raw_text: RawTextFactory
    ) -> None:
        invoice = make_invoice(line_items=(), tax_breakdown=TaxBreakdown())
        report = review(invoice, make_raw_text(invoice))

        assert report.decision is ApprovalDecision.AUTO_APPROVE
        subtotal_checks = next(
            fc.checks for fc in report.field_confidences if fc.field is InvoiceField.SUBTOTAL
        )
        assert all(check.category is not CheckCategory.ARITHMETIC for check in subtotal_checks)


def test_empty_confidence_set_is_untrusted_and_held() -> None:
    assert overall_confidence(()) == 0.0
    assert decide((), DEFAULT_POLICY) is ApprovalDecision.HOLD


def test_absent_text_fields_are_skipped_not_flagged(
    make_invoice: InvoiceFactory, make_raw_text: RawTextFactory
) -> None:
    # A receipt with no vendor / number / date / GSTIN must not be penalised for
    # fields it simply does not carry — those checks skip, like an absent GSTIN.
    invoice = make_invoice(
        vendor_name="", invoice_number="", invoice_date="", currency="", vendor_gstin=None
    )
    report = review(invoice, make_raw_text(invoice))

    reported = {fc.field for fc in report.field_confidences}
    assert InvoiceField.INVOICE_DATE not in reported
    assert InvoiceField.GSTIN not in reported
    assert InvoiceField.CURRENCY not in reported
    assert report.decision is ApprovalDecision.AUTO_APPROVE


def test_non_integral_amounts_are_cross_checked() -> None:
    invoice = build_invoice(
        line_items=(),
        tax_breakdown=TaxBreakdown(),
        subtotal=Money.of("156100.50"),
        tax=Money.of("28099.50"),
        total=Money.of("184200.00"),
    )
    report = review(invoice, build_raw_text(invoice))

    assert report.decision is ApprovalDecision.AUTO_APPROVE


def test_invalid_invoice_number_is_flagged(
    make_invoice: InvoiceFactory, make_raw_text: RawTextFactory
) -> None:
    invoice = make_invoice(invoice_number="THIS-NUMBER-IS-TOO-LONG")
    report = review(invoice, make_raw_text(invoice))

    assert any(
        flag.category is CheckCategory.FORMAT and flag.field is InvoiceField.INVOICE_NUMBER
        for flag in report.flags
    )


def test_single_non_critical_failure_routes_to_human_review(
    make_invoice: InvoiceFactory, make_raw_text: RawTextFactory
) -> None:
    # An unknown currency is a format failure on a non-critical field: confidence
    # drops to the review band rather than triggering an outright hold.
    invoice = make_invoice(currency="XYZ")
    report = review(invoice, make_raw_text(invoice))

    assert report.decision is ApprovalDecision.HUMAN_REVIEW


class TestValueObjectAccessors:
    def test_money_multiplies_by_a_factor(self) -> None:
        assert Money.of("78050.00") * 2 == Money.of("156100.00")

    def test_gstin_stringifies_to_its_value(self) -> None:
        base = "29AAGCB1234M1Z"
        gstin = Gstin(base + Gstin.compute_check_digit(base))
        assert str(gstin) == gstin.value

    def test_invoice_number_stringifies_to_its_value(self) -> None:
        assert str(InvoiceNumber("SE-9921")) == "SE-9921"
