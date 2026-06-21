from __future__ import annotations

from apverify.domain.critique import InvoiceField
from apverify.infrastructure.mapping import (
    ConfidentInvoiceDTO,
    FieldConfidenceDTO,
    InvoiceDTO,
    to_confident_extraction,
    to_domain,
)

_BASE = {
    "vendor_name": "ACME Steel Pvt Ltd",
    "invoice_number": "INV-1",
    "invoice_date": "04-06-2025",
    "subtotal": "100",
    "tax": "18",
    "total": "118",
}


def test_verbalized_confidence_maps_onto_invoice_fields() -> None:
    dto = ConfidentInvoiceDTO(
        **_BASE,
        field_confidence=FieldConfidenceDTO(vendor_name=0.9, total=0.4),
    )

    extraction = to_confident_extraction(dto)

    assert extraction.invoice.vendor_name == "ACME Steel Pvt Ltd"
    assert extraction.confidences[InvoiceField.VENDOR] == 0.9
    assert extraction.confidences[InvoiceField.TOTAL] == 0.4


def test_confidence_is_clamped_into_the_unit_interval() -> None:
    dto = ConfidentInvoiceDTO(
        **_BASE,
        field_confidence=FieldConfidenceDTO.model_validate({"total": 1.4, "tax": -0.2}),
    )

    extraction = to_confident_extraction(dto)

    assert extraction.confidences[InvoiceField.TOTAL] == 1.0
    assert extraction.confidences[InvoiceField.TAX] == 0.0


def test_confidence_defaults_to_zero_when_the_model_omits_it() -> None:
    extraction = to_confident_extraction(ConfidentInvoiceDTO(**_BASE))

    assert extraction.confidences[InvoiceField.CURRENCY] == 0.0


def test_null_identity_fields_become_empty_rather_than_rejecting_the_invoice() -> None:
    # a provider that emits null for an unlabelled invoice number must not fail the whole
    # extraction — the empty value is later read as "absent" by the critic.
    payload = {**_BASE, "invoice_number": None, "invoice_date": None}
    invoice = to_domain(InvoiceDTO.model_validate(payload))

    assert invoice.invoice_number == ""
    assert invoice.invoice_date == ""


def test_null_subtotal_falls_back_to_total() -> None:
    payload = {**_BASE, "subtotal": None, "total": "118"}
    invoice = to_domain(InvoiceDTO.model_validate(payload))

    assert str(invoice.subtotal) == "118.00"
    assert str(invoice.total) == "118.00"


def test_null_total_becomes_zero_rather_than_rejecting_the_invoice() -> None:
    payload = {**_BASE, "total": None}
    invoice = to_domain(InvoiceDTO.model_validate(payload))

    assert str(invoice.total) == "0.00"


def test_null_line_item_cells_do_not_reject_the_extraction() -> None:
    # a provider that leaves a line's price or quantity null must not fail the invoice.
    null_line = {"description": None, "quantity": None, "unit_price": None, "line_total": None}
    payload = {**_BASE, "line_items": [null_line]}
    invoice = to_domain(InvoiceDTO.model_validate(payload))

    line = invoice.line_items[0]
    assert line.quantity == 1
    assert str(line.unit_price) == "0.00"
    assert line.description == ""
