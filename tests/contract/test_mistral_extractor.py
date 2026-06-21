"""Live contract test for the Mistral (Pixtral) extractor.

Skipped unless MISTRAL_API_KEY is set, since it makes a real, billable API call.
Verifies the adapter honours the ``InvoiceExtractor`` port: a rendered page in, a
domain ``Invoice`` out.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SAMPLE = Path(__file__).resolve().parents[2] / "samples" / "clean_invoice_01.pdf"


@pytest.mark.skipif(not os.getenv("MISTRAL_API_KEY"), reason="no MISTRAL_API_KEY in environment")
def test_mistral_extracts_a_domain_invoice() -> None:
    from mistralai import Mistral

    from apverify.domain.invoice import Invoice
    from apverify.infrastructure.errors import RenderError
    from apverify.infrastructure.mistral.extractor import MistralInvoiceExtractor
    from apverify.infrastructure.rendering.pdf import Pdf2ImageRenderer

    try:
        pages = Pdf2ImageRenderer(dpi=200).render(_SAMPLE)
    except RenderError as exc:
        pytest.skip(f"poppler not available: {exc}")

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    extractor = MistralInvoiceExtractor([client], os.getenv("MISTRAL_MODEL", "pixtral-12b-2409"))

    invoice = extractor.extract(pages)

    assert isinstance(invoice, Invoice)
    assert invoice.total.amount > 0
