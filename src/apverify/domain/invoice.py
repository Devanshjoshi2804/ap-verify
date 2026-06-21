"""Invoice entities.

These hold the data exactly as a model extracted it — including values that may
be wrong. Catching wrong-but-well-formed data is the critic's job, so the entity
deliberately stores raw strings for fields whose validity is itself a finding
(GSTIN, invoice number, date) rather than refusing to exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from apverify.domain.value_objects import Money


@dataclass(frozen=True, slots=True)
class LineItem:
    description: str
    quantity: int
    unit_price: Money
    line_total: Money
    hsn_sac: str | None = None


@dataclass(frozen=True, slots=True)
class TaxBreakdown:
    """Indian GST splits tax into CGST+SGST (intra-state) or IGST (inter-state)."""

    cgst: Money | None = None
    sgst: Money | None = None
    igst: Money | None = None

    def components(self) -> tuple[Money, ...]:
        return tuple(c for c in (self.cgst, self.sgst, self.igst) if c is not None)


@dataclass(frozen=True, slots=True)
class Invoice:
    vendor_name: str
    invoice_number: str
    invoice_date: str
    currency: str
    subtotal: Money
    tax: Money
    total: Money
    line_items: tuple[LineItem, ...] = ()
    tax_breakdown: TaxBreakdown = field(default_factory=TaxBreakdown)
    vendor_gstin: str | None = None
    bank_account: str | None = None
    purchase_order_ref: str | None = None
