from __future__ import annotations

from pathlib import Path

import pytest

from apverify.infrastructure.errors import RenderError
from apverify.infrastructure.rendering.pdf import Pdf2ImageRenderer

pytestmark = pytest.mark.contract

_SAMPLE = Path(__file__).resolve().parents[2] / "samples" / "clean_invoice_01.pdf"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_renderer_rasterises_a_pdf_into_png_pages() -> None:
    renderer = Pdf2ImageRenderer(dpi=150)
    try:
        pages = renderer.render(_SAMPLE)
    except RenderError as exc:
        pytest.skip(f"poppler not available: {exc}")

    assert len(pages) >= 1
    assert all(page.media_type == "image/png" for page in pages)
    assert all(page.data.startswith(_PNG_MAGIC) for page in pages)
