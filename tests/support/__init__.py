"""Shared test support: a deterministic invoice factory, matching OCR text, and
in-memory fakes for the three ports.

The fakes are hand-written test doubles, not mocking-library magic — they
implement the port protocols structurally, which keeps the application and CLI
tests honest about the actual interfaces.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

from apverify.application.errors import PortError
from apverify.application.ports import PageImage
from apverify.domain.audit import AuditVerdict
from apverify.domain.collections import Receivable
from apverify.domain.critique import InvoiceField
from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.ocr import RawText
from apverify.domain.procurement import (
    GoodsReceiptLine,
    GoodsReceiptNote,
    PurchaseOrder,
    PurchaseOrderLine,
)
from apverify.domain.value_objects import Gstin, Money, PhoneNumber

InvoiceFactory = Callable[..., Invoice]
RawTextFactory = Callable[..., RawText]

_GSTIN_BASE = "27AABCU9603R1Z"
VALID_GSTIN = _GSTIN_BASE + Gstin.compute_check_digit(_GSTIN_BASE)
PO_NUMBER = "PO-2025-1001"


def build_invoice(**overrides: object) -> Invoice:
    defaults: dict[str, object] = {
        "vendor_name": "ACME Steel Pvt Ltd",
        "invoice_number": "INV-2025-0042",
        "invoice_date": "04-06-2025",
        "currency": "INR",
        "subtotal": Money.of("156100.00"),
        "tax": Money.of("28100.00"),
        "total": Money.of("184200.00"),
        "line_items": (
            LineItem("TMT Steel Bars (12mm)", 2, Money.of("78050.00"), Money.of("156100.00")),
        ),
        "tax_breakdown": TaxBreakdown(cgst=Money.of("14050.00"), sgst=Money.of("14050.00")),
        "vendor_gstin": VALID_GSTIN,
    }
    defaults.update(overrides)
    return Invoice(**defaults)  # type: ignore[arg-type]


def build_raw_text(invoice: Invoice, *, omit: Sequence[str] = ()) -> RawText:
    """Page text that faithfully contains the invoice's values, minus any omitted.

    The total is written with Indian-style separators to exercise the
    normalisation in :meth:`RawText.contains`.
    """
    fragments: dict[str, str] = {
        "vendor_name": invoice.vendor_name,
        "vendor_gstin": invoice.vendor_gstin or "",
        "invoice_number": invoice.invoice_number,
        "invoice_date": invoice.invoice_date,
    }
    lines = [value for key, value in fragments.items() if key not in omit and value]
    lines.append(f"Subtotal {invoice.subtotal}")
    lines.extend(_tax_lines(invoice))
    lines.append(f"Total INR {invoice.total.amount:,.2f}")
    lines.extend(f"{item.description} {item.line_total}" for item in invoice.line_items)
    return RawText(text="\n".join(lines))


def build_purchase_order(**overrides: object) -> PurchaseOrder:
    defaults: dict[str, object] = {
        "po_number": PO_NUMBER,
        "vendor_name": "ACME Steel Pvt Ltd",
        "currency": "INR",
        "subtotal": Money.of("156100.00"),
        "lines": (PurchaseOrderLine("TMT Steel Bars (12mm)", 2, Money.of("78050.00")),),
        "vendor_gstin": VALID_GSTIN,
    }
    defaults.update(overrides)
    return PurchaseOrder(**defaults)  # type: ignore[arg-type]


def build_goods_receipt(**overrides: object) -> GoodsReceiptNote:
    defaults: dict[str, object] = {
        "grn_number": "GRN-77",
        "purchase_order_ref": PO_NUMBER,
        "lines": (GoodsReceiptLine("TMT Steel Bars (12mm)", 2),),
    }
    defaults.update(overrides)
    return GoodsReceiptNote(**defaults)  # type: ignore[arg-type]


def _tax_lines(invoice: Invoice) -> list[str]:
    """Mirror a real GST invoice: print CGST/SGST/IGST separately, never the sum."""
    breakdown = invoice.tax_breakdown
    labelled = (("CGST", breakdown.cgst), ("SGST", breakdown.sgst), ("IGST", breakdown.igst))
    printed = [f"{label} {amount}" for label, amount in labelled if amount is not None]
    return printed or [f"Tax {invoice.tax}"]


class FakeRenderer:
    def __init__(self, pages: list[PageImage] | None = None) -> None:
        self._pages = pages if pages is not None else [PageImage(data=b"page")]

    def render(self, document: object) -> list[PageImage]:
        return self._pages


class FakeExtractor:
    def __init__(self, invoice: Invoice) -> None:
        self._invoice = invoice

    def extract(self, pages: Sequence[PageImage]) -> Invoice:
        return self._invoice


class FakeOcr:
    def __init__(self, raw_text: RawText) -> None:
        self._raw_text = raw_text

    def read(self, pages: Sequence[PageImage]) -> RawText:
        return self._raw_text


class FailingExtractor:
    """An extractor that always fails — for testing graceful degradation."""

    def extract(self, pages: Sequence[PageImage]) -> Invoice:
        raise PortError("simulated extractor failure")


def build_receivable(**overrides: object) -> Receivable:
    defaults: dict[str, object] = {
        "customer_name": "Acme Buyer Ltd",
        "phone": PhoneNumber("+919812345678"),
        "invoice_number": "AR-2025-0001",
        "amount_due": Money.of("84500.00"),
        "currency": "INR",
        "due_date": date(2026, 6, 1),
    }
    defaults.update(overrides)
    return Receivable(**defaults)  # type: ignore[arg-type]


class RecordingSender:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, phone: str, text: str) -> str:
        self.sent.append((phone, text))
        return "wamid.TEST"


class FailingSender:
    def send(self, phone: str, text: str) -> str:
        raise PortError("whatsapp unreachable")


class FakeAuditor:
    def __init__(self, verdicts: Sequence[AuditVerdict] = ()) -> None:
        self._verdicts = tuple(verdicts)
        self.audited_fields: tuple[InvoiceField, ...] = ()

    def audit(
        self, invoice: Invoice, raw_text: RawText, fields: Sequence[InvoiceField]
    ) -> list[AuditVerdict]:
        self.audited_fields = tuple(fields)
        return list(self._verdicts)
