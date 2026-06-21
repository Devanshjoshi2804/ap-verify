from __future__ import annotations

from collections.abc import Mapping, Sequence

from apverify.application.ports import ConfidentExtraction, PageImage
from apverify.domain.critique import InvoiceField
from apverify.domain.invoice import Invoice
from apverify.domain.ocr import RawText
from apverify.domain.value_objects import Money
from apverify.eval.accuracy_eval import LabelledDocument
from apverify.eval.calibration_eval import collect_uncertainty_samples


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


class _FakeExtractor:
    def __init__(self, invoice: Invoice, confidences: Mapping[InvoiceField, float]) -> None:
        self._extraction = ConfidentExtraction(invoice=invoice, confidences=confidences)

    def extract_with_confidence(self, pages: Sequence[PageImage]) -> ConfidentExtraction:
        return self._extraction


def test_collects_both_signals_against_the_same_truth() -> None:
    invoice = _invoice(vendor="ACME Steel Pvt Ltd", total="118")
    document = LabelledDocument(
        label="doc",
        truth={"vendor": "ACME Steel Pvt Ltd", "total": "118"},
        pages=(PageImage(data=b"img"),),
        raw_text=RawText("ACME Steel Pvt Ltd Total 118.00"),
    )
    extractor = _FakeExtractor(invoice, {InvoiceField.VENDOR: 0.95, InvoiceField.TOTAL: 0.6})

    samples = collect_uncertainty_samples([document], extractor)

    # Both correct against truth, so every verbalized sample pairs with True.
    assert {c for c, _ in samples.verbalized} == {0.95, 0.6}
    assert all(correct for _, correct in samples.verbalized)
    assert samples.critic  # critic produced its own confidences for the same fields


def test_verbalized_confidence_tracks_a_wrong_extraction() -> None:
    invoice = _invoice(vendor="ACME Steel Pvt Ltd", total="999")  # total is wrong
    document = LabelledDocument(
        label="doc",
        truth={"total": "118"},
        pages=(PageImage(data=b"img"),),
        raw_text=RawText("Total 118.00"),
    )
    extractor = _FakeExtractor(invoice, {InvoiceField.TOTAL: 0.3})

    samples = collect_uncertainty_samples([document], extractor)

    assert samples.verbalized == [(0.3, False)]
