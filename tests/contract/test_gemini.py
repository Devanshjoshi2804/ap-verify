"""Live contract test for the Gemini extractor.

Skipped unless GEMINI_API_KEY is set, since it makes a real, billable API call.
It verifies that the adapter honours the ``InvoiceExtractor`` port end to end: a
rendered page in, a domain ``Invoice`` out.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SAMPLE = Path(__file__).resolve().parents[2] / "samples" / "clean_invoice_01.pdf"


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="no GEMINI_API_KEY in environment")
def test_gemini_extracts_a_domain_invoice() -> None:
    from google import genai

    from apverify.domain.invoice import Invoice
    from apverify.infrastructure.errors import RenderError
    from apverify.infrastructure.gemini.extractor import GeminiInvoiceExtractor
    from apverify.infrastructure.rendering.pdf import Pdf2ImageRenderer

    try:
        pages = Pdf2ImageRenderer(dpi=200).render(_SAMPLE)
    except RenderError as exc:
        pytest.skip(f"poppler not available: {exc}")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    extractor = GeminiInvoiceExtractor([client], os.getenv("GEMINI_MODEL", "gemini-flash-latest"))

    invoice = extractor.extract(pages)

    assert isinstance(invoice, Invoice)
    assert invoice.total.amount > 0
