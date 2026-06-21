from __future__ import annotations

from apverify.infrastructure.mapping import InvoiceDTO, LineItemDTO, to_domain


def _dto(**over: object) -> InvoiceDTO:
    base: dict[str, object] = {
        "vendor_name": "ACME",
        "invoice_number": "INV-1",
        "invoice_date": "04-06-2025",
        "currency": "INR",
        "total": "100",
    }
    base.update(over)
    return InvoiceDTO.model_validate(base)


def test_non_numeric_line_item_amount_becomes_zero() -> None:
    # A model put a unit-of-measure ("Pkg.") where a price belongs — must not crash.
    dto = _dto(line_items=[LineItemDTO(description="x", unit_price="Pkg.", line_total="Pkg.")])
    invoice = to_domain(dto)
    assert str(invoice.line_items[0].unit_price) == "0.00"
    assert str(invoice.line_items[0].line_total) == "0.00"


def test_non_numeric_invoice_total_falls_back_rather_than_crashing() -> None:
    invoice = to_domain(_dto(total="N/A"))
    assert str(invoice.total) == "0.00"


def test_empty_quantity_coerces_to_one_rather_than_crashing() -> None:
    # A model left the quantity cell blank ("") — must default to 1, not fail the
    # whole extraction (three real DocILE pages were lost to this).
    dto = LineItemDTO(description="x", quantity="", unit_price="10", line_total="10")
    assert dto.quantity == 1


def test_decimal_quantity_truncates_rather_than_crashing() -> None:
    # A model returned a measure ("87.3") where a whole count belongs.
    dto = LineItemDTO(description="x", quantity="87.3", unit_price="10", line_total="10")
    assert dto.quantity == 87


def test_literal_null_vendor_is_treated_as_absent() -> None:
    # A model emitted the string "null" for the vendor; the critic must read that as a
    # missing field, not a vendor literally named "null".
    invoice = to_domain(_dto(vendor_name="null"))
    assert invoice.vendor_name == ""
