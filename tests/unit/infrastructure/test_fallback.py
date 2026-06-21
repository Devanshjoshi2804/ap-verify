from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from apverify.application.ports import ConfidentExtraction, PageImage
from apverify.domain.critique import InvoiceField
from apverify.domain.invoice import Invoice
from apverify.domain.value_objects import Money
from apverify.infrastructure.errors import ExtractionError
from apverify.infrastructure.fallback import FallbackInvoiceExtractor

_PAGES = [PageImage(data=b"img")]


def _invoice(vendor: str) -> Invoice:
    return Invoice(
        vendor_name=vendor,
        invoice_number="INV-1",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=Money.of("100"),
        tax=Money.of("18"),
        total=Money.of("118"),
    )


class _Exhausted:
    def extract(self, pages: Sequence[PageImage]) -> Invoice:
        raise ExtractionError("quota")

    def extract_with_confidence(self, pages: Sequence[PageImage]) -> ConfidentExtraction:
        raise ExtractionError("quota")


class _Working:
    def __init__(
        self, vendor: str, confidences: Mapping[InvoiceField, float] | None = None
    ) -> None:
        self._invoice = _invoice(vendor)
        self._confidences = confidences or {}

    def extract(self, pages: Sequence[PageImage]) -> Invoice:
        return self._invoice

    def extract_with_confidence(self, pages: Sequence[PageImage]) -> ConfidentExtraction:
        return ConfidentExtraction(invoice=self._invoice, confidences=self._confidences)


class _PlainOnly:
    """A provider that cannot report verbalized confidence."""

    def extract(self, pages: Sequence[PageImage]) -> Invoice:
        return _invoice("Plain Co")


def test_fallback_advances_past_an_exhausted_provider() -> None:
    extractor = FallbackInvoiceExtractor([_Exhausted(), _Working("Second Co")])
    assert extractor.extract(_PAGES).vendor_name == "Second Co"


def test_fallback_raises_only_when_every_provider_fails() -> None:
    extractor = FallbackInvoiceExtractor([_Exhausted(), _Exhausted()])
    with pytest.raises(ExtractionError, match="all 2 providers failed"):
        extractor.extract(_PAGES)


def test_fallback_routes_confidence_to_a_capable_provider() -> None:
    extractor = FallbackInvoiceExtractor(
        [_Exhausted(), _Working("Second Co", {InvoiceField.TOTAL: 0.7})]
    )
    result = extractor.extract_with_confidence(_PAGES)
    assert result.confidences[InvoiceField.TOTAL] == 0.7


def test_fallback_skips_providers_that_cannot_report_confidence() -> None:
    # the plain provider is not even consulted for confidence; the capable one answers
    extractor = FallbackInvoiceExtractor([_PlainOnly(), _Working("Conf Co")])
    assert extractor.extract_with_confidence(_PAGES).invoice.vendor_name == "Conf Co"


def test_fallback_confidence_fails_when_no_provider_supports_it() -> None:
    extractor = FallbackInvoiceExtractor([_PlainOnly()])
    with pytest.raises(ExtractionError, match="verbalized confidence"):
        extractor.extract_with_confidence(_PAGES)


def test_fallback_requires_at_least_one_extractor() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FallbackInvoiceExtractor([])
