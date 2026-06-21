from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw, ImageFont

from apverify.application.ports import PageImage
from apverify.infrastructure.errors import OcrError
from apverify.infrastructure.ocr.tesseract import TesseractOcrProvider

pytestmark = pytest.mark.contract


def _page_showing(text: str) -> PageImage:
    image = Image.new("RGB", (900, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 70), text, fill="black", font=ImageFont.load_default(size=64))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return PageImage(data=buffer.getvalue())


def test_tesseract_reads_a_page_and_supports_the_cross_check() -> None:
    provider = TesseractOcrProvider(min_confidence=0.0)
    try:
        raw_text = provider.read([_page_showing("TOTAL 184200")])
    except OcrError as exc:
        pytest.skip(f"tesseract not available: {exc}")

    assert raw_text.contains("184200")
    assert raw_text.words
