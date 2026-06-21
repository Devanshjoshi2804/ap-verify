"""Groq vision adapter — a third, independent ``InvoiceExtractor``.

Groq serves an open vision model (Llama-4 Scout) on a separate free tier, so it
backs the primary when its quota is spent and, being a different vendor and
architecture, doubles as another cross-model opinion. Reuses the same ``InvoiceDTO``
schema and domain mapping as the other extractors, so its output is validated and
compared on identical terms.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import TypeVar

from groq import APIError, Groq
from pydantic import BaseModel, ValidationError

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
    "Extract the invoice fields from this image as JSON matching the schema "
    "{vendor_name, invoice_number, invoice_date, currency, subtotal, tax, total, "
    "cgst, sgst, igst, vendor_gstin, purchase_order_ref, line_items:[{description, "
    "quantity, unit_price, line_total, hsn_sac}]}. Amounts as digit strings without "
    "separators. currency as the ISO 4217 code. Use null for anything absent. "
    + VENDOR_GUIDANCE
    + " "
    + LINE_ITEM_GUIDANCE
    + " Return JSON only."
)

_DTO = TypeVar("_DTO", bound=BaseModel)


class GroqInvoiceExtractor:
    def __init__(self, clients: Sequence[Groq], model: str) -> None:
        if not clients:
            raise ValueError("at least one Groq client is required")
        self._clients = tuple(clients)
        self._model = model

    def extract(self, pages: Sequence[PageImage]) -> Invoice:
        return to_domain(self._complete(pages, _PROMPT, InvoiceDTO))

    def extract_with_confidence(self, pages: Sequence[PageImage]) -> ConfidentExtraction:
        prompt = _PROMPT + CONFIDENCE_INSTRUCTION
        return to_confident_extraction(self._complete(pages, prompt, ConfidentInvoiceDTO))

    def extract_samples(self, pages: Sequence[PageImage], samples: int) -> tuple[Invoice, ...]:
        return tuple(
            to_domain(self._complete(pages, _PROMPT, InvoiceDTO, temperature=0.7))
            for _ in range(samples)
        )

    def _complete(
        self, pages: Sequence[PageImage], prompt: str, schema: type[_DTO], temperature: float = 0.0
    ) -> _DTO:
        if not pages:
            raise ExtractionError("no pages to extract from")

        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        content += [{"type": "image_url", "image_url": {"url": _data_uri(page)}} for page in pages]

        last_error: Exception | None = None
        for client in self._clients:
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": content}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                )
                payload = response.choices[0].message.content
                if not isinstance(payload, str):
                    raise ExtractionError("Groq did not return a JSON string")
                return schema.model_validate_json(payload)
            except (APIError, ValidationError, ExtractionError) as exc:
                last_error = exc

        raise ExtractionError(
            f"Groq extraction failed across {len(self._clients)} key(s): {last_error}"
        ) from last_error


def _data_uri(page: PageImage) -> str:
    encoded = base64.b64encode(page.data).decode("ascii")
    return f"data:{page.media_type};base64,{encoded}"
