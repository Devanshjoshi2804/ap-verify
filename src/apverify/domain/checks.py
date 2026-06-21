"""The critic's verification rules — pure functions over a domain ``Invoice`` and
the OCR ``RawText``.

Three independent layers, cheapest first:

* **OCR cross-check** — does each reported value actually appear on the page?
  A value the OCR never saw was invented by the extractor.
* **Arithmetic** — do the line items, subtotal, tax and total reconcile? A single
  misread digit breaks the sum even when every value looks individually plausible.
* **Format** — does the GSTIN pass its checksum, the date parse, the currency
  resolve? Structural validity the model has no incentive to guarantee.

Nothing here touches I/O, so the entire critic is unit-testable with hand-built
invoices and no model, OCR engine, or fixtures.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal

from apverify.domain.critique import (
    DEFAULT_POLICY,
    CheckCategory,
    CheckResult,
    CheckStatus,
    CriticReport,
    FieldConfidence,
    InvoiceField,
    Policy,
    decide,
    overall_confidence,
    score_field,
)
from apverify.domain.errors import InvalidGstinError, InvalidInvoiceNumberError
from apverify.domain.invoice import Invoice
from apverify.domain.ocr import RawText
from apverify.domain.value_objects import Gstin, InvoiceNumber, Money

_ACCEPTED_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m/%d/%y",
    "%d/%m/%y",
    "%d %b %Y",
    "%d %B %Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%b %d %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)
_ACCEPTED_CURRENCIES = frozenset({"INR", "USD", "EUR", "GBP", "AED", "SGD", "IDR"})

_FIELD_ORDER = (
    InvoiceField.VENDOR,
    InvoiceField.GSTIN,
    InvoiceField.INVOICE_NUMBER,
    InvoiceField.INVOICE_DATE,
    InvoiceField.CURRENCY,
    InvoiceField.LINE_ITEMS,
    InvoiceField.SUBTOTAL,
    InvoiceField.TAX,
    InvoiceField.TOTAL,
)


def review(invoice: Invoice, raw_text: RawText, policy: Policy = DEFAULT_POLICY) -> CriticReport:
    """Run every check and roll the findings up into a decision."""
    results = [
        *_cross_checks(invoice, raw_text),
        *_arithmetic_checks(invoice, policy),
        *_format_checks(invoice),
    ]

    grouped: dict[InvoiceField, list[CheckResult]] = defaultdict(list)
    for result in results:
        grouped[result.field].append(result)

    field_confidences = tuple(
        FieldConfidence(
            field=field,
            value=_display_value(invoice, field),
            confidence=score_field(tuple(checks), policy),
            checks=tuple(checks),
        )
        for field in _FIELD_ORDER
        if (checks := grouped.get(field)) and _has_signal(checks)
    )

    return CriticReport(
        field_confidences=field_confidences,
        overall_confidence=overall_confidence(field_confidences),
        decision=decide(field_confidences, policy),
    )


def _cross_checks(invoice: Invoice, raw_text: RawText) -> list[CheckResult]:
    checks = [
        _cross_check_vendor(invoice.vendor_name, raw_text),
        _cross_check_amount(InvoiceField.SUBTOTAL, invoice.subtotal, raw_text),
        _cross_check_amount(InvoiceField.TOTAL, invoice.total, raw_text),
        _cross_check_text(InvoiceField.INVOICE_NUMBER, invoice.invoice_number, raw_text),
        _cross_check_text(InvoiceField.GSTIN, invoice.vendor_gstin, raw_text),
    ]
    checks.extend(_tax_cross_checks(invoice, raw_text))
    return checks


def _tax_cross_checks(invoice: Invoice, raw_text: RawText) -> list[CheckResult]:
    """Cross-check the figures that are actually printed.

    A GST invoice prints CGST/SGST/IGST as separate lines, not their sum, so we
    verify each component against the page. The combined ``tax`` is left to the
    arithmetic check, which ties the components to the total.
    """
    components = invoice.tax_breakdown.components()
    if components:
        return [_cross_check_amount(InvoiceField.TAX, amount, raw_text) for amount in components]
    return [_cross_check_amount(InvoiceField.TAX, invoice.tax, raw_text)]


def _cross_check_amount(field: InvoiceField, amount: Money, raw_text: RawText) -> CheckResult:
    found = any(raw_text.contains(form) for form in _amount_candidates(amount))
    return _result(
        CheckCategory.CROSS_CHECK,
        field,
        passed=found,
        detail=(
            f"{amount} found in page text" if found else f"{amount} does not appear on the page"
        ),
    )


def _cross_check_vendor(vendor_name: str, raw_text: RawText) -> CheckResult:
    if not vendor_name.strip():
        return _skipped(CheckCategory.CROSS_CHECK, InvoiceField.VENDOR, "no vendor extracted")
    # Vendor names are multi-word and OCR-mangled; match on token overlap, not an
    # exact substring, so a faithfully-read vendor isn't flagged as hallucinated.
    found = raw_text.contains_most_tokens(vendor_name)
    return _result(
        CheckCategory.CROSS_CHECK,
        InvoiceField.VENDOR,
        passed=found,
        detail=f"{vendor_name!r} {'corroborated by' if found else 'not found in'} page text",
    )


def _cross_check_text(field: InvoiceField, value: str | None, raw_text: RawText) -> CheckResult:
    if value is None or not value.strip():
        return _skipped(CheckCategory.CROSS_CHECK, field, "no value extracted")
    found = raw_text.contains(value)
    return _result(
        CheckCategory.CROSS_CHECK,
        field,
        passed=found,
        detail=f"{value!r} {'found' if found else 'not found'} in page text",
    )


def _arithmetic_checks(invoice: Invoice, policy: Policy) -> list[CheckResult]:
    tol = policy.arithmetic_tolerance
    checks: list[CheckResult] = []

    if invoice.line_items:
        line_sum = _sum_money(item.line_total for item in invoice.line_items)
        checks.append(
            _result(
                CheckCategory.ARITHMETIC,
                InvoiceField.SUBTOTAL,
                passed=_close(line_sum, invoice.subtotal, tol),
                detail=f"Σ line totals {line_sum} vs subtotal {invoice.subtotal}",
            )
        )
        # A non-zero line with a zero unit price is impossible — it means the unit price
        # was unreadable (a junk value coerced to 0 at the boundary). Surface it rather
        # than let a confident 0 stand in for an extraction we could not actually read.
        unreadable = sum(
            1
            for item in invoice.line_items
            if item.unit_price.amount == 0 and item.line_total.amount != 0
        )
        if unreadable:
            checks.append(
                _result(
                    CheckCategory.ARITHMETIC,
                    InvoiceField.SUBTOTAL,
                    passed=False,
                    detail=f"{unreadable} line item(s) have a zero unit price on a "
                    "non-zero line — an unreadable amount, not a real value",
                )
            )

    components = invoice.tax_breakdown.components()
    if components:
        component_sum = _sum_money(components)
        checks.append(
            _result(
                CheckCategory.ARITHMETIC,
                InvoiceField.TAX,
                passed=_close(component_sum, invoice.tax, tol),
                detail=f"CGST+SGST+IGST {component_sum} vs tax {invoice.tax}",
            )
        )

    expected_total = invoice.subtotal + invoice.tax
    checks.append(
        _result(
            CheckCategory.ARITHMETIC,
            InvoiceField.TOTAL,
            passed=_close(expected_total, invoice.total, tol),
            detail=f"subtotal+tax {expected_total} vs total {invoice.total}",
        )
    )
    return checks


def _format_checks(invoice: Invoice) -> list[CheckResult]:
    return [
        _format_gstin(invoice.vendor_gstin),
        _format_invoice_number(invoice.invoice_number),
        _format_date(invoice.invoice_date),
        _format_currency(invoice.currency),
    ]


def _format_gstin(value: str | None) -> CheckResult:
    if value is None or not value.strip():
        return _skipped(CheckCategory.FORMAT, InvoiceField.GSTIN, "no GSTIN extracted")
    try:
        Gstin(value)
    except InvalidGstinError as exc:
        return _result(CheckCategory.FORMAT, InvoiceField.GSTIN, passed=False, detail=str(exc))
    return _result(
        CheckCategory.FORMAT, InvoiceField.GSTIN, passed=True, detail="format and checksum valid"
    )


def _format_invoice_number(value: str) -> CheckResult:
    if not value.strip():
        return _skipped(CheckCategory.FORMAT, InvoiceField.INVOICE_NUMBER, "no number extracted")
    try:
        InvoiceNumber(value)
    except InvalidInvoiceNumberError as exc:
        return _result(
            CheckCategory.FORMAT, InvoiceField.INVOICE_NUMBER, passed=False, detail=str(exc)
        )
    return _result(
        CheckCategory.FORMAT, InvoiceField.INVOICE_NUMBER, passed=True, detail="within GST rules"
    )


def _format_date(value: str) -> CheckResult:
    if not value.strip():
        return _skipped(CheckCategory.FORMAT, InvoiceField.INVOICE_DATE, "no date extracted")
    parsed = _parse_date(value)
    return _result(
        CheckCategory.FORMAT,
        InvoiceField.INVOICE_DATE,
        passed=parsed,
        detail=f"{value!r} {'parses as a date' if parsed else 'is not a recognisable date'}",
    )


def _format_currency(value: str) -> CheckResult:
    if not value.strip():
        return _skipped(CheckCategory.FORMAT, InvoiceField.CURRENCY, "no currency extracted")
    valid = value.strip().upper() in _ACCEPTED_CURRENCIES
    return _result(
        CheckCategory.FORMAT,
        InvoiceField.CURRENCY,
        passed=valid,
        detail=f"{value!r} {'is a known currency' if valid else 'is not a known currency'}",
    )


def parse_date(value: str) -> date | None:
    """The date in ``value`` under any accepted format, or ``None`` if none parse."""
    for fmt in _ACCEPTED_DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_date(value: str) -> bool:
    return parse_date(value) is not None


def _amount_candidates(amount: Money) -> list[str]:
    """Printed forms an amount might take, so ``184200.00`` matches a page that
    prints ``1,84,200`` and one that prints ``1,84,200.00``."""
    forms = [f"{amount.amount:.2f}"]
    if amount.amount == amount.amount.to_integral_value():
        forms.append(f"{amount.amount.to_integral_value()}")
    return forms


def _sum_money(amounts: Iterable[Money]) -> Money:
    total = Money.of(0)
    for amount in amounts:
        total += amount
    return total


def _close(left: Money, right: Money, tolerance: Decimal) -> bool:
    return abs(left.amount - right.amount) <= tolerance


def _display_value(invoice: Invoice, field: InvoiceField) -> str:
    values: dict[InvoiceField, object] = {
        InvoiceField.VENDOR: invoice.vendor_name,
        InvoiceField.GSTIN: invoice.vendor_gstin,
        InvoiceField.INVOICE_NUMBER: invoice.invoice_number,
        InvoiceField.INVOICE_DATE: invoice.invoice_date,
        InvoiceField.CURRENCY: invoice.currency,
        InvoiceField.LINE_ITEMS: f"{len(invoice.line_items)} line(s)",
        InvoiceField.SUBTOTAL: invoice.subtotal,
        InvoiceField.TAX: invoice.tax,
        InvoiceField.TOTAL: invoice.total,
    }
    return str(values.get(field, ""))


def _has_signal(checks: list[CheckResult]) -> bool:
    return any(check.status is not CheckStatus.SKIPPED for check in checks)


def _result(
    category: CheckCategory, field: InvoiceField, *, passed: bool, detail: str
) -> CheckResult:
    status = CheckStatus.PASSED if passed else CheckStatus.FAILED
    return CheckResult(category=category, field=field, status=status, detail=detail)


def _skipped(category: CheckCategory, field: InvoiceField, detail: str) -> CheckResult:
    return CheckResult(category=category, field=field, status=CheckStatus.SKIPPED, detail=detail)
