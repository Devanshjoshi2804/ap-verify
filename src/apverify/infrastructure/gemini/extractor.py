"""Gemini vision adapter implementing the ``InvoiceExtractor`` port.

Image-native extraction (the model sees the page, not just OCR text) handles
messy layouts that text-only parsing trips on. Structured output bound to
``InvoiceDTO`` makes schema validation the first, cheapest hallucination guard:
malformed output is rejected before it reaches the critic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from apverify.application.ports import ConfidentExtraction, PageImage
from apverify.domain.invoice import Invoice
from apverify.infrastructure.errors import ExtractionError
from apverify.infrastructure.mapping import (
    CONFIDENCE_INSTRUCTION,
    LINE_ITEM_GUIDANCE,
    VENDOR_GUIDANCE,
    ConfidentInvoiceDTO,
    InvoiceDTO,
    to_confident_extraction,
    to_domain,
)

_PROMPT = (
    """You are extracting fields from a single supplier invoice image.
Return only what is printed on the page. Never guess or complete a value you
cannot read — use null for anything absent or illegible.

Rules:
- Amounts: digits only, no thousands separators or currency symbols (e.g. "184200.00").
- currency: the ISO 4217 code (INR, USD, ...).
- vendor_gstin: the 15-character GSTIN exactly as printed, else null.
- invoice_date: copy the date string as shown on the invoice.
- purchase_order_ref: the PO number the invoice cites (e.g. "PO-..."), else null.
- line_items: one entry per row, with its own line_total as printed.
- """
    + VENDOR_GUIDANCE
    + "\n- "
    + LINE_ITEM_GUIDANCE
)

_DTO = TypeVar("_DTO", bound=BaseModel)


class GeminiInvoiceExtractor:
    def __init__(self, clients: Sequence[genai.Client], model: str) -> None:
        if not clients:
            raise ValueError("at least one Gemini client is required")
        self._clients = tuple(clients)
        self._model = model

    def extract(self, pages: Sequence[PageImage]) -> Invoice:
        return to_domain(self._generate(pages, _PROMPT, InvoiceDTO))

    def extract_with_confidence(self, pages: Sequence[PageImage]) -> ConfidentExtraction:
        prompt = _PROMPT + CONFIDENCE_INSTRUCTION
        return to_confident_extraction(self._generate(pages, prompt, ConfidentInvoiceDTO))

    def extract_samples(self, pages: Sequence[PageImage], samples: int) -> tuple[Invoice, ...]:
        # Non-zero temperature so the draws actually vary — the spread is the signal.
        return tuple(
            to_domain(self._generate(pages, _PROMPT, InvoiceDTO, temperature=0.7))
            for _ in range(samples)
        )

    def _generate(
        self, pages: Sequence[PageImage], prompt: str, schema: type[_DTO], temperature: float = 0.0
    ) -> _DTO:
        if not pages:
            raise ExtractionError("no pages to extract from")

        contents = [
            types.Part.from_bytes(data=page.data, mime_type=page.media_type) for page in pages
        ]
        contents.append(types.Part.from_text(text=prompt))
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        )

        # Try each key in turn: a rate-limited (429) key rotates to the next.
        last_error: Exception | None = None
        for client in self._clients:
            try:
                response = client.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
                dto = response.parsed
                if not isinstance(dto, schema):
                    raise ExtractionError("model did not return a structured invoice")
                return dto
            except (ExtractionError, genai_errors.APIError) as exc:
                last_error = exc

        raise ExtractionError(
            f"extraction failed across {len(self._clients)} key(s): {last_error}"
        ) from last_error
