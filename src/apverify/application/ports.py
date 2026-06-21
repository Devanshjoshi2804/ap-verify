"""Ports the use case depends on, defined as structural ``Protocol``s.

The application owns these interfaces; infrastructure implements them. Pages cross
the boundary as encoded image bytes rather than a library-specific image object,
so neither the use case nor the domain ever imports PIL, the Gemini SDK, or
Tesseract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from apverify.domain.anomaly import AnomalyAssessment
from apverify.domain.audit import AuditVerdict
from apverify.domain.collections import Receivable
from apverify.domain.critique import InvoiceField
from apverify.domain.fraud import IdentifiedInvoice
from apverify.domain.invoice import Invoice
from apverify.domain.ocr import RawText
from apverify.domain.procurement import GoodsReceiptNote, PurchaseOrder
from apverify.domain.vendor_master import KnownVendor


@dataclass(frozen=True, slots=True)
class PageImage:
    data: bytes
    media_type: str = "image/png"


class DocumentRenderer(Protocol):
    def render(self, document: Path) -> list[PageImage]:
        """Rasterise a PDF/image file into one encoded image per page."""
        ...


class InvoiceExtractor(Protocol):
    def extract(self, pages: Sequence[PageImage]) -> Invoice:
        """Read the invoice fields from the page image(s)."""
        ...


@dataclass(frozen=True, slots=True)
class ConfidentExtraction:
    """An extraction paired with the model's own 0..1 confidence per field.

    This is *verbalized* confidence — what the model says about itself — kept
    separate from the entity it produced. Useful only to calibration, never to the
    approval pipeline, so it lives beside the port rather than on ``Invoice``.
    """

    invoice: Invoice
    confidences: Mapping[InvoiceField, float]


@runtime_checkable
class SelfReportingExtractor(Protocol):
    def extract_with_confidence(self, pages: Sequence[PageImage]) -> ConfidentExtraction:
        """Extract the invoice and ask the model how sure it is of each field."""
        ...


@runtime_checkable
class SamplingExtractor(Protocol):
    def extract_samples(self, pages: Sequence[PageImage], samples: int) -> tuple[Invoice, ...]:
        """Extract the invoice ``samples`` times at non-zero temperature, for
        sampling-based uncertainty (self-consistency / semantic entropy)."""
        ...


class OcrTextProvider(Protocol):
    def read(self, pages: Sequence[PageImage]) -> RawText:
        """Return the raw OCR text and word boxes for the page image(s)."""
        ...


class ProcurementRepository(Protocol):
    def purchase_order_for(self, invoice: Invoice) -> PurchaseOrder | None:
        """The purchase order the invoice cites, if one is on file."""
        ...

    def goods_receipt_for(self, purchase_order: PurchaseOrder) -> GoodsReceiptNote | None:
        """The goods-receipt note recorded against a purchase order, if any."""
        ...


class SemanticAuditor(Protocol):
    def audit(
        self, invoice: Invoice, raw_text: RawText, fields: Sequence[InvoiceField]
    ) -> list[AuditVerdict]:
        """Judge whether the given fields' values are supported by the document."""
        ...


class MessageSender(Protocol):
    def send(self, phone: str, text: str) -> str:
        """Send a message and return the provider's message id."""
        ...


class ReceivablesRepository(Protocol):
    def list_receivables(self) -> list[Receivable]:
        """All outstanding receivables to consider for collection."""
        ...


@runtime_checkable
class InvoiceLedger(Protocol):
    """Source of previously-seen invoices to check a candidate against for duplicates.

    Defined here as the integration seam; the duplicate benchmark supplies priors
    directly, and a production adapter (a persisted store of posted invoices) will
    implement this when the fraud stage is wired into the pipeline.
    """

    def known_invoices(self) -> tuple[IdentifiedInvoice, ...]: ...


@runtime_checkable
class VendorMasterRepository(Protocol):
    """The roster of known vendors and their established bank accounts, checked against
    an incoming invoice for bank-change / impersonation (BEC) risk."""

    def known_vendors(self) -> tuple[KnownVendor, ...]: ...


@runtime_checkable
class AnomalyDetector(Protocol):
    """Scores how statistically unusual an invoice is for its vendor, given history."""

    def score(self, invoice: Invoice, history: Sequence[Invoice]) -> AnomalyAssessment: ...


@runtime_checkable
class VendorHistoryRepository(Protocol):
    """A vendor's previously-seen invoices, the baseline an anomaly is measured against."""

    def history_for(self, invoice: Invoice) -> tuple[Invoice, ...]: ...
