"""Tesseract adapter implementing the ``OcrTextProvider`` port.

Tesseract is the second, independent reader of the page. It is intentionally a
different engine from the vision model: agreement between two unrelated readers is
what makes the cross-check meaningful. ``image_to_data`` gives per-word boxes and
confidences; we keep words at level 5 above a confidence floor.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import pytesseract
from PIL import Image, UnidentifiedImageError
from pytesseract import Output

from apverify.application.ports import PageImage
from apverify.domain.ocr import RawText, WordBox
from apverify.infrastructure.errors import OcrError

_WORD_LEVEL = 5


class TesseractOcrProvider:
    def __init__(self, min_confidence: float = 60.0, language: str = "eng") -> None:
        self._min_confidence = min_confidence
        self._language = language

    def read(self, pages: Sequence[PageImage]) -> RawText:
        texts: list[str] = []
        words: list[WordBox] = []
        vertical_offset = 0

        for page in pages:
            image = self._open(page)
            try:
                texts.append(pytesseract.image_to_string(image, lang=self._language))
                data = pytesseract.image_to_data(
                    image, lang=self._language, output_type=Output.DICT
                )
            except pytesseract.TesseractNotFoundError as exc:
                raise OcrError("the tesseract binary is not installed or not on PATH") from exc
            words.extend(self._words(data, vertical_offset))
            vertical_offset += image.height

        return RawText(text="\n".join(texts), words=tuple(words))

    def _open(self, page: PageImage) -> Image.Image:
        try:
            return Image.open(io.BytesIO(page.data))
        except UnidentifiedImageError as exc:
            raise OcrError("page image could not be decoded") from exc

    def _words(self, data: Any, vertical_offset: int) -> list[WordBox]:
        boxes: list[WordBox] = []
        for index in range(len(data["text"])):
            if int(data["level"][index]) != _WORD_LEVEL:
                continue
            confidence = float(data["conf"][index])
            text = str(data["text"][index]).strip()
            if not text or confidence < self._min_confidence:
                continue
            left = int(data["left"][index])
            top = int(data["top"][index]) + vertical_offset
            width = int(data["width"][index])
            height = int(data["height"][index])
            boxes.append(
                WordBox(
                    text=text,
                    left=left,
                    top=top,
                    right=left + width,
                    bottom=top + height,
                    confidence=confidence,
                )
            )
        return boxes
