"""Generate the sample invoices used by the demo and the contract tests.

Deterministic by design: no randomness, valid GSTINs computed from the domain's
own checksum, and one copy with a single digit flipped in the total so the critic
has something real to catch. Run with the project venv active:

    python samples/generate_samples.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from apverify.domain.value_objects import Gstin


@dataclass(frozen=True)
class _Line:
    description: str
    quantity: int
    unit_price: int


@dataclass(frozen=True)
class _Invoice:
    filename: str
    vendor: str
    address: str
    gstin_base: str
    number: str
    date: str
    lines: tuple[_Line, ...]
    cgst: int
    sgst: int
    po_ref: str
    total_override: int | None = field(default=None)

    @property
    def gstin(self) -> str:
        return self.gstin_base + Gstin.compute_check_digit(self.gstin_base)

    @property
    def subtotal(self) -> int:
        return sum(line.quantity * line.unit_price for line in self.lines)

    @property
    def tax(self) -> int:
        return self.cgst + self.sgst

    @property
    def printed_total(self) -> int:
        return self.total_override if self.total_override is not None else self.subtotal + self.tax


_INVOICES = (
    _Invoice(
        filename="clean_invoice_01.pdf",
        vendor="ACME Steel Pvt Ltd",
        address="Plot 14, MIDC Industrial Area, Pune 411019",
        gstin_base="27AABCU9603R1Z",
        number="INV-2025-0042",
        date="04-06-2025",
        lines=(_Line("TMT Steel Bars (12mm)", 2, 78050),),
        cgst=14050,
        sgst=14050,
        po_ref="PO-2025-1001",
    ),
    _Invoice(
        filename="clean_invoice_02.pdf",
        vendor="Bharat Textiles LLP",
        address="221 Ring Road, Surat 395002",
        gstin_base="29AAGCB1234M1Z",
        number="BT/2025/318",
        date="17-05-2025",
        lines=(
            _Line("Cotton Fabric (roll)", 12, 4200),
            _Line("Polyester Blend (roll)", 8, 3600),
        ),
        cgst=7092,
        sgst=7092,
        po_ref="PO-2025-1002",
    ),
    _Invoice(
        filename="clean_invoice_03.pdf",
        vendor="Sunrise Electronics",
        address="5 Nehru Place, New Delhi 110019",
        gstin_base="07AAECS5678K1Z",
        number="SE-9921",
        date="29-04-2025",
        lines=(
            _Line("LED Panel 40W", 25, 1450),
            _Line("Driver Module", 25, 320),
        ),
        cgst=3982,
        sgst=3982,
        po_ref="PO-2025-1003",
    ),
)

# A faithful copy of invoice 01 whose printed total has one digit flipped
# (1,84,200 -> 1,84,000): the page is internally inconsistent, so the arithmetic
# check must reject it even though every value is legible.
_CORRUPTED = _Invoice(
    filename="corrupted_total.pdf",
    vendor=_INVOICES[0].vendor,
    address=_INVOICES[0].address,
    gstin_base=_INVOICES[0].gstin_base,
    number=_INVOICES[0].number,
    date=_INVOICES[0].date,
    lines=_INVOICES[0].lines,
    cgst=_INVOICES[0].cgst,
    sgst=_INVOICES[0].sgst,
    po_ref=_INVOICES[0].po_ref,
    total_override=184000,
)


def _rupees(amount: int) -> str:
    return f"INR {amount:,.2f}"


def _draw(invoice: _Invoice, destination: Path) -> None:
    pdf = canvas.Canvas(str(destination), pagesize=A4)
    _, height = A4
    left = 20 * mm
    cursor = height - 25 * mm

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(left, cursor, invoice.vendor)
    cursor -= 7 * mm
    pdf.setFont("Helvetica", 10)
    pdf.drawString(left, cursor, invoice.address)
    cursor -= 5 * mm
    pdf.drawString(left, cursor, f"GSTIN: {invoice.gstin}")
    cursor -= 10 * mm

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(left, cursor, "TAX INVOICE")
    cursor -= 6 * mm
    pdf.setFont("Helvetica", 10)
    pdf.drawString(left, cursor, f"Invoice No: {invoice.number}")
    pdf.drawString(left + 90 * mm, cursor, f"Date: {invoice.date}")
    cursor -= 5 * mm
    pdf.drawString(left, cursor, f"PO Ref: {invoice.po_ref}")
    cursor -= 9 * mm

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(left, cursor, "Description")
    pdf.drawString(left + 90 * mm, cursor, "Qty")
    pdf.drawString(left + 110 * mm, cursor, "Rate")
    pdf.drawString(left + 150 * mm, cursor, "Amount")
    cursor -= 6 * mm
    pdf.setFont("Helvetica", 10)
    for line in invoice.lines:
        pdf.drawString(left, cursor, line.description)
        pdf.drawString(left + 90 * mm, cursor, str(line.quantity))
        pdf.drawString(left + 110 * mm, cursor, _rupees(line.unit_price))
        pdf.drawString(left + 150 * mm, cursor, _rupees(line.quantity * line.unit_price))
        cursor -= 6 * mm

    cursor -= 4 * mm
    for label, amount in (
        ("Subtotal", invoice.subtotal),
        ("CGST", invoice.cgst),
        ("SGST", invoice.sgst),
        ("Total", invoice.printed_total),
    ):
        pdf.setFont("Helvetica-Bold" if label == "Total" else "Helvetica", 10)
        pdf.drawString(left + 110 * mm, cursor, label)
        pdf.drawString(left + 150 * mm, cursor, _rupees(amount))
        cursor -= 6 * mm

    pdf.showPage()
    pdf.save()


def main() -> None:
    output_dir = Path(__file__).parent
    for invoice in (*_INVOICES, _CORRUPTED):
        _draw(invoice, output_dir / invoice.filename)
        print(f"wrote {invoice.filename}")


if __name__ == "__main__":
    main()
