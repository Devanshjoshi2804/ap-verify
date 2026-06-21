"""The wire schema the vision model fills in, and its translation to the domain.

Keeping a separate DTO means the model's JSON shape — amounts as strings, a flat
tax breakdown, optional fields — never leaks inward. The domain receives a proper
``Invoice`` of value objects; if the model returns garbage, conversion raises an
``ExtractionError`` here at the boundary rather than corrupting the core.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field, ValidationError, field_validator

from apverify.application.ports import ConfidentExtraction
from apverify.domain.critique import InvoiceField
from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.value_objects import Money
from apverify.infrastructure.errors import ExtractionError


def _parseable_amount(value: object) -> bool:
    try:
        Decimal(str(value).replace(",", "").strip())
    except (ArithmeticError, ValueError):
        return False
    return True


def _stringify_amount(value: object) -> object:
    """Coerce a numeric amount to a string at the wire boundary, or ``None`` when the
    model put something unparseable (e.g. a unit-of-measure) where an amount belongs —
    so the downstream fallbacks (null total -> 0, null subtotal -> total) handle it
    rather than the whole extraction failing over one junk cell.

    Models that honour our schema return amount strings; those that only support
    generic JSON return numbers. Normalising keeps the path into ``Money`` float-free.
    """
    if value is None:
        return None
    stringified = str(value) if isinstance(value, (int, float)) else value
    return stringified if _parseable_amount(stringified) else None


def _amount_or_zero(value: object) -> object:
    """A line-item amount, treating a model's null *or unreadable* cell as zero rather
    than failing the whole extraction over it."""
    coerced = _stringify_amount(value)
    return "0" if coerced is None else coerced


def _quantity_or_one(value: object) -> object:
    """A line-item quantity, defaulting to 1 when the model omits it or returns
    something that isn't a whole count — a blank cell or a decimal measure (e.g.
    "87.3") — rather than failing the whole extraction over one cell."""
    if value is None:
        return 1
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return 1


# Whole-string values a model emits to mean "no value"; treated as an absent field
# rather than a literal name so the critic reads them as missing.
_NULLISH = frozenset({"", "null", "none", "n/a", "nan"})


def _blank_if_nullish(value: object) -> object:
    if isinstance(value, str) and value.strip().lower() in _NULLISH:
        return ""
    return "" if value is None else value


class LineItemDTO(BaseModel):
    description: str = ""
    quantity: int = 1
    unit_price: str
    line_total: str
    hsn_sac: str | None = None

    _coerce = field_validator("unit_price", "line_total", mode="before")(_amount_or_zero)
    _qty = field_validator("quantity", mode="before")(_quantity_or_one)
    _desc = field_validator("description", mode="before")(lambda v: "" if v is None else v)


class InvoiceDTO(BaseModel):
    """Amounts are strings so no float ever sits between the model and ``Money``."""

    vendor_name: str
    invoice_number: str
    invoice_date: str
    currency: str = "INR"
    subtotal: str | None = None  # some models leave it blank; falls back to total below
    tax: str | None = None  # some models leave the combined tax blank; derived below
    total: str | None = None
    line_items: list[LineItemDTO] = Field(default_factory=list)
    cgst: str | None = None
    sgst: str | None = None
    igst: str | None = None
    vendor_gstin: str | None = None
    bank_account: str | None = None
    purchase_order_ref: str | None = None

    _coerce = field_validator("subtotal", "tax", "total", "cgst", "sgst", "igst", mode="before")(
        _stringify_amount
    )
    # Some models emit null for an absent identity field; treat it as empty (the critic
    # already reads an empty field as "not present") rather than rejecting the whole
    # extraction over one missing label.
    _absent = field_validator("vendor_name", "invoice_number", "invoice_date", mode="before")(
        _blank_if_nullish
    )


def _clamp_unit(value: object) -> object:
    """Pin a self-reported confidence into ``[0, 1]``; models drift outside it."""
    if isinstance(value, (int, float)):
        return min(max(float(value), 0.0), 1.0)
    return value


class FieldConfidenceDTO(BaseModel):
    """The model's own 0..1 confidence for each scorable field."""

    vendor_name: float = 0.0
    invoice_date: float = 0.0
    currency: float = 0.0
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0

    _clamp = field_validator(
        "vendor_name", "invoice_date", "currency", "subtotal", "tax", "total", mode="before"
    )(_clamp_unit)


class ConfidentInvoiceDTO(InvoiceDTO):
    """An invoice plus the model's per-field self-assessment."""

    field_confidence: FieldConfidenceDTO = Field(default_factory=FieldConfidenceDTO)


# Embedded in every provider's prompt so the seller-vs-buyer distinction is stated
# identically everywhere. On real invoices the model often returns the buyer — the
# advertiser or customer the bill is addressed to — instead of the party that issued
# it; this pins vendor_name to the payee.
VENDOR_GUIDANCE = (
    "vendor_name is the SELLER that issued this invoice and is owed payment — the "
    "party in the 'from' / 'remit to' position, usually at the top. It is NOT the "
    "buyer, customer, recipient, or advertiser the invoice is billed to. Use the "
    "seller's full legal entity name exactly as printed, including any suffix such "
    "as Inc, LLC or Ltd."
)


# Embedded in every provider's prompt. Models tend to collapse a run of identical
# rows (same description and/or amount) into one line and silently sum them; on
# itemised media/utility invoices that loses most of the table. This pins one entry
# per printed row.
LINE_ITEM_GUIDANCE = (
    "Emit one line_items entry for every printed row, in order. Even when rows repeat "
    "with the same description or amount, keep them as separate entries — never merge, "
    "sum, or deduplicate rows. Each entry's line_total is that single row as printed."
)


# Appended to an extractor's prompt to elicit verbalized confidence. Shared so every
# provider asks for it in identical terms.
CONFIDENCE_INSTRUCTION = (
    "\nAlso fill field_confidence: for each of vendor_name, invoice_date, currency, "
    "subtotal, tax and total, your own 0..1 confidence that the value you returned is "
    "correct. Be honest and discriminating — use a low score when the page is unclear "
    "or the value was hard to read, near 1.0 only when it is plainly legible."
)


def to_confident_extraction(dto: ConfidentInvoiceDTO) -> ConfidentExtraction:
    confidence = dto.field_confidence
    return ConfidentExtraction(
        invoice=to_domain(dto),
        confidences={
            InvoiceField.VENDOR: confidence.vendor_name,
            InvoiceField.INVOICE_DATE: confidence.invoice_date,
            InvoiceField.CURRENCY: confidence.currency,
            InvoiceField.SUBTOTAL: confidence.subtotal,
            InvoiceField.TAX: confidence.tax,
            InvoiceField.TOTAL: confidence.total,
        },
    )


def to_domain(dto: InvoiceDTO) -> Invoice:
    try:
        breakdown = TaxBreakdown(
            cgst=_optional_money(dto.cgst),
            sgst=_optional_money(dto.sgst),
            igst=_optional_money(dto.igst),
        )
        # An absent total is read as zero (arithmetic will flag it); an absent subtotal
        # falls back to the total, so a single missing amount never fails the mapping.
        total = Money.of(dto.total) if dto.total is not None else Money.of(0)
        subtotal = Money.of(dto.subtotal) if dto.subtotal is not None else total
        return Invoice(
            vendor_name=dto.vendor_name,
            invoice_number=dto.invoice_number,
            invoice_date=dto.invoice_date,
            currency=dto.currency,
            subtotal=subtotal,
            tax=Money.of(dto.tax) if dto.tax is not None else _sum(breakdown.components()),
            total=total,
            line_items=tuple(_line_item(item) for item in dto.line_items),
            tax_breakdown=breakdown,
            vendor_gstin=dto.vendor_gstin,
            bank_account=dto.bank_account,
            purchase_order_ref=dto.purchase_order_ref,
        )
    except (ValueError, ValidationError) as exc:
        raise ExtractionError(f"model output could not be mapped to an invoice: {exc}") from exc


def _line_item(dto: LineItemDTO) -> LineItem:
    return LineItem(
        description=dto.description,
        quantity=dto.quantity,
        unit_price=Money.of(dto.unit_price),
        line_total=Money.of(dto.line_total),
        hsn_sac=dto.hsn_sac,
    )


def _optional_money(value: str | None) -> Money | None:
    return Money.of(value) if value is not None else None


def _sum(amounts: tuple[Money, ...]) -> Money:
    total = Money.of(0)
    for amount in amounts:
        total += amount
    return total
