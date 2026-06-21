from __future__ import annotations

from collections.abc import Mapping, Sequence

from apverify.application.ports import ConfidentExtraction, PageImage
from apverify.domain.critique import InvoiceField
from apverify.domain.invoice import Invoice
from apverify.domain.ocr import RawText
from apverify.domain.value_objects import Money
from apverify.eval.accuracy_eval import LabelledDocument
from apverify.eval.fusion_eval import collect_feature_rows


def _invoice(vendor: str, total: str) -> Invoice:
    return Invoice(
        vendor_name=vendor,
        invoice_number="INV-1",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=Money.of("100"),
        tax=Money.of("18"),
        total=Money.of(total),
    )


class _Primary:
    def __init__(self, invoice: Invoice, confidences: Mapping[InvoiceField, float]) -> None:
        self._extraction = ConfidentExtraction(invoice=invoice, confidences=confidences)

    def extract_with_confidence(self, pages: Sequence[PageImage]) -> ConfidentExtraction:
        return self._extraction


class _Secondary:
    def __init__(self, invoice: Invoice) -> None:
        self._invoice = invoice

    def extract(self, pages: Sequence[PageImage]) -> Invoice:
        return self._invoice


def _document() -> LabelledDocument:
    return LabelledDocument(
        label="doc",
        truth={"vendor": "ACME Steel Pvt Ltd", "total": "118"},
        pages=(PageImage(data=b"img"),),
        raw_text=RawText("ACME Steel Pvt Ltd Total 118.00"),
    )


def test_cross_model_agreement_becomes_a_feature() -> None:
    invoice = _invoice("ACME Steel Pvt Ltd", "118")
    primary = _Primary(invoice, {InvoiceField.VENDOR: 0.9, InvoiceField.TOTAL: 0.8})
    rows = collect_feature_rows([_document()], primary, _Secondary(invoice))

    total = next(r for r in rows if r.field == "total")
    assert total.cross_model_agrees is True
    assert total.correct is True
    assert total.verbalized_confidence == 0.8


def test_second_model_disagreement_flags_a_confidently_wrong_field() -> None:
    primary_invoice = _invoice("ACME Steel Pvt Ltd", "999")  # total wrong, both extractors
    secondary_invoice = _invoice("ACME Steel Pvt Ltd", "118")  # second model reads it right
    primary = _Primary(primary_invoice, {InvoiceField.TOTAL: 0.95})
    rows = collect_feature_rows([_document()], primary, _Secondary(secondary_invoice))

    total = next(r for r in rows if r.field == "total")
    assert total.correct is False  # primary was wrong
    assert total.cross_model_agrees is False  # and the second model disagrees -> caught


def test_document_is_dropped_when_the_second_model_fails() -> None:
    from apverify.application.errors import PortError

    class _Broken:
        def extract(self, pages: Sequence[PageImage]) -> Invoice:
            raise PortError("down")

    invoice = _invoice("ACME Steel Pvt Ltd", "118")
    primary = _Primary(invoice, {InvoiceField.TOTAL: 0.9})
    assert collect_feature_rows([_document()], primary, _Broken()) == []
