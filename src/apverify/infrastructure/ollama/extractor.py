"""Ollama adapter — a local, unlimited ``InvoiceExtractor``.

Its value here is not accuracy (a small local model reads worse than a hosted one)
but *independence with no quota*: an extractor that fails in different places than
the primary, callable an unbounded number of times. That makes it the natural second
leg for the fusion cross-model signal — disagreement between two independently-wrong
models is what flags the confidently-wrong extractions nothing else catches — and it
removes the rate-limit bottleneck from accumulating fusion rows.

Talks to Ollama's native ``/api/chat`` with base64 images and ``format: json``. Reuses
the shared ``InvoiceDTO``/``ConfidentInvoiceDTO`` (and their null tolerance), so its
output is validated and compared on identical terms.

Pages are downscaled before sending. A 300-DPI A4 scan tiles into thousands of vision
tokens, which a 7B model on a workstation GPU/CPU cannot hold — it OOM-kills mid-decode.
Capping the long edge keeps the page legible while bounding the tile count; the hosted
extractors keep full resolution, so only the resource-constrained local leg pays for it.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import TypeVar

import httpx
from PIL import Image
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

# Long-edge cap (px). Keeps a full A4 page legible while bounding the local model's
# vision-token count to a few tiles — enough to read, small enough not to OOM a 7B.
_MAX_EDGE = 1400


class OllamaInvoiceExtractor:
    def __init__(self, client: httpx.Client, model: str) -> None:
        self._client = client
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

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [_encode_downscaled(page) for page in pages],
                }
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            return schema.model_validate_json(content)
        except (httpx.HTTPError, ValidationError, KeyError, ValueError) as exc:
            raise ExtractionError(f"Ollama extraction failed: {exc}") from exc


def _encode_downscaled(page: PageImage) -> str:
    """Shrink the page to ``_MAX_EDGE`` on its long side, re-encode, base64."""
    image = Image.open(io.BytesIO(page.data))
    image.thumbnail((_MAX_EDGE, _MAX_EDGE))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
