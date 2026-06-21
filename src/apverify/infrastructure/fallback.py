"""Provider-level fallback — the same idea as multi-key rotation, one layer up.

A single vision provider's free tier runs out; an independent one (different vendor,
different quota) usually has not. This composite chains extractors and advances to
the next whenever one fails — quota, transport, or unusable output — so a run only
fails if *every* provider is down. It satisfies both the plain extractor port and the
verbalized-confidence port, delegating confidence to whichever chained providers can
report it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from apverify.application.ports import (
    ConfidentExtraction,
    InvoiceExtractor,
    PageImage,
    SamplingExtractor,
    SelfReportingExtractor,
)
from apverify.domain.invoice import Invoice
from apverify.infrastructure.errors import AdapterError, ExtractionError

_T = TypeVar("_T")
_R = TypeVar("_R")


class FallbackInvoiceExtractor:
    def __init__(self, extractors: Sequence[InvoiceExtractor]) -> None:
        if not extractors:
            raise ValueError("at least one extractor is required")
        self._extractors = tuple(extractors)

    def extract(self, pages: Sequence[PageImage]) -> Invoice:
        return _first_success(self._extractors, lambda extractor: extractor.extract(pages))

    def extract_with_confidence(self, pages: Sequence[PageImage]) -> ConfidentExtraction:
        capable = tuple(e for e in self._extractors if isinstance(e, SelfReportingExtractor))
        if not capable:
            raise ExtractionError("no chained provider can report verbalized confidence")
        return _first_success(capable, lambda extractor: extractor.extract_with_confidence(pages))

    def extract_samples(self, pages: Sequence[PageImage], samples: int) -> tuple[Invoice, ...]:
        capable = tuple(e for e in self._extractors if isinstance(e, SamplingExtractor))
        if not capable:
            raise ExtractionError("no chained provider can resample for uncertainty")
        return _first_success(capable, lambda extractor: extractor.extract_samples(pages, samples))


def _first_success(providers: Sequence[_T], call: Callable[[_T], _R]) -> _R:
    last_error: AdapterError | None = None
    for provider in providers:
        try:
            return call(provider)
        except AdapterError as exc:
            last_error = exc
    raise ExtractionError(f"all {len(providers)} providers failed: {last_error}") from last_error
