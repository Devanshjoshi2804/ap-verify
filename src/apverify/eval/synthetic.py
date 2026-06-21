"""Deterministic synthetic invoices with perfect ground truth.

We control both the extracted ``Invoice`` and the OCR ``RawText`` a faithful
reader would produce, so the labels are exact and free — no dataset access, no
licensing, no annotation. That lets the harness measure the critic in isolation:
a corrupted copy reviewed against the *correct* page text is exactly the
hallucination scenario we care about.

No randomness (it would break reproducibility); every value is derived from the
row index.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.ocr import RawText
from apverify.domain.value_objects import Gstin, Money

_VENDORS: tuple[tuple[str, str], ...] = (
    ("ACME Steel Pvt Ltd", "27AABCU9603R1Z"),
    ("Bharat Textiles LLP", "29AAGCB1234M1Z"),
    ("Sunrise Electronics", "07AAECS5678K1Z"),
    ("Deccan Polymers", "36AADCD2244P1Z"),
    ("Konkan Foods Pvt Ltd", "24AAFCK8890Q1Z"),
)
_PRODUCTS = (
    "TMT Steel Bars (12mm)",
    "Cotton Fabric (roll)",
    "LED Panel 40W",
    "HDPE Granules (bag)",
    "Refined Oil (tin)",
)
_GST_RATE = Decimal("0.09")  # CGST and SGST, 9% each (18% total)


@dataclass(frozen=True, slots=True)
class GroundTruth:
    label: str
    invoice: Invoice


def generate_dataset(count: int = 25) -> list[GroundTruth]:
    return [_build(index) for index in range(count)]


def faithful_raw_text(invoice: Invoice) -> RawText:
    """The page text a perfect OCR pass over this invoice would yield."""
    lines = [
        invoice.vendor_name,
        f"GSTIN {invoice.vendor_gstin}",
        f"Invoice No {invoice.invoice_number}",
        f"Date {invoice.invoice_date}",
        f"PO Ref {invoice.purchase_order_ref}",
    ]
    lines += [
        f"{item.description}  {item.quantity}  {item.line_total}" for item in invoice.line_items
    ]
    lines.append(f"Subtotal {invoice.subtotal}")
    breakdown = invoice.tax_breakdown
    lines += [
        f"{name} {amount}"
        for name, amount in (("CGST", breakdown.cgst), ("SGST", breakdown.sgst))
        if amount
    ]
    lines.append(f"Total INR {invoice.total.amount:,.2f}")
    return RawText(text="\n".join(lines))


def _build(index: int) -> GroundTruth:
    vendor, gstin_base = _VENDORS[index % len(_VENDORS)]
    gstin = gstin_base + Gstin.compute_check_digit(gstin_base)

    quantity = 2 + index % 5
    unit_price = Money.of(1500 + (index * 137) % 8000)
    line_total = unit_price * quantity
    subtotal = line_total
    cgst = Money((subtotal.amount * _GST_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    tax = cgst + cgst
    total = subtotal + tax

    invoice = Invoice(
        vendor_name=vendor,
        invoice_number=f"INV-2025-{1000 + index}",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=subtotal,
        tax=tax,
        total=total,
        line_items=(LineItem(_PRODUCTS[index % len(_PRODUCTS)], quantity, unit_price, line_total),),
        tax_breakdown=TaxBreakdown(cgst=cgst, sgst=cgst),
        vendor_gstin=gstin,
        purchase_order_ref=f"PO-2025-{2000 + index}",
    )
    return GroundTruth(label=f"synthetic-{index:03d}", invoice=invoice)
