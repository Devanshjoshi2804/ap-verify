"""The one use case in v0: turn a document into a reviewed invoice.

Orchestration only — it wires the ports together and hands the result to the
domain critic. No SDK, no framework, no I/O of its own; that all lives behind the
injected ports, which is what makes this testable with in-memory fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apverify.application.ports import DocumentRenderer, InvoiceExtractor, OcrTextProvider
from apverify.domain.checks import review
from apverify.domain.critique import DEFAULT_POLICY, CriticReport, Policy
from apverify.domain.invoice import Invoice
from apverify.domain.ocr import RawText


@dataclass(frozen=True, slots=True)
class InvoiceReview:
    invoice: Invoice
    raw_text: RawText
    report: CriticReport


class ProcessInvoiceUseCase:
    def __init__(
        self,
        renderer: DocumentRenderer,
        extractor: InvoiceExtractor,
        ocr: OcrTextProvider,
        policy: Policy = DEFAULT_POLICY,
    ) -> None:
        self._renderer = renderer
        self._extractor = extractor
        self._ocr = ocr
        self._policy = policy

    def execute(self, document: Path) -> InvoiceReview:
        pages = self._renderer.render(document)
        invoice = self._extractor.extract(pages)
        raw_text = self._ocr.read(pages)
        report = review(invoice, raw_text, self._policy)
        return InvoiceReview(invoice=invoice, raw_text=raw_text, report=report)
