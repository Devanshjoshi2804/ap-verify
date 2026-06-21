from __future__ import annotations

import io
import json

import httpx
import pytest
from PIL import Image

from apverify.application.ports import PageImage
from apverify.domain.critique import InvoiceField
from apverify.infrastructure.errors import ExtractionError
from apverify.infrastructure.ollama.extractor import OllamaInvoiceExtractor


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


_PAGES = [PageImage(data=_png_bytes())]
_INVOICE_JSON = json.dumps(
    {
        "vendor_name": "ACME Steel Pvt Ltd",
        "invoice_number": "INV-1",
        "invoice_date": "04-06-2025",
        "subtotal": "100",
        "tax": "18",
        "total": "118",
        "field_confidence": {"vendor_name": 0.8, "total": 0.6},
    }
)


def _extractor(handler: object) -> OllamaInvoiceExtractor:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    client = httpx.Client(transport=transport, base_url="http://localhost:11434")
    return OllamaInvoiceExtractor(client, "qwen2.5vl:7b")


def test_ollama_extracts_from_a_local_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["messages"][0]["images"]  # the page was sent as a base64 image
        return httpx.Response(200, json={"message": {"content": _INVOICE_JSON}})

    invoice = _extractor(handler).extract(_PAGES)
    assert invoice.vendor_name == "ACME Steel Pvt Ltd"
    assert str(invoice.total) == "118.00"


def test_ollama_reports_verbalized_confidence() -> None:
    extraction = _extractor(
        lambda _r: httpx.Response(200, json={"message": {"content": _INVOICE_JSON}})
    ).extract_with_confidence(_PAGES)
    assert extraction.confidences[InvoiceField.VENDOR] == 0.8


def test_ollama_failure_becomes_extraction_error_so_the_chain_can_advance() -> None:
    def refused(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ExtractionError, match="Ollama extraction failed"):
        _extractor(refused).extract(_PAGES)
