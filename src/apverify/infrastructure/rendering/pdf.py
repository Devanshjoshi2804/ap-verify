"""Document renderer implementing the ``DocumentRenderer`` port.

Both the vision model and the OCR engine work on raster images, so PDFs are
rasterised (via poppler) and image files are normalised to PNG. Page images are
the single representation handed to both downstream readers, guaranteeing they
look at exactly the same pixels.
"""

from __future__ import annotations

import io
from pathlib import Path

from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
from PIL import Image, UnidentifiedImageError

from apverify.application.ports import PageImage
from apverify.infrastructure.errors import RenderError

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})


class Pdf2ImageRenderer:
    def __init__(self, dpi: int = 300) -> None:
        self._dpi = dpi

    def render(self, document: Path) -> list[PageImage]:
        if not document.exists():
            raise RenderError(f"document not found: {document}")

        suffix = document.suffix.lower()
        if suffix == ".pdf":
            images = self._render_pdf(document)
        elif suffix in _IMAGE_SUFFIXES:
            images = [self._load_image(document)]
        else:
            raise RenderError(f"unsupported document type: {suffix or '(none)'}")

        return [self._encode(image) for image in images]

    def _render_pdf(self, document: Path) -> list[Image.Image]:
        try:
            return convert_from_path(str(document), dpi=self._dpi)
        except (PDFInfoNotInstalledError, PDFPageCountError) as exc:
            raise RenderError(
                f"could not rasterise the PDF; is poppler installed? ({type(exc).__name__})"
            ) from exc

    def _load_image(self, document: Path) -> Image.Image:
        try:
            return Image.open(document).convert("RGB")
        except UnidentifiedImageError as exc:
            raise RenderError(f"could not read image: {document}") from exc

    def _encode(self, image: Image.Image) -> PageImage:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return PageImage(data=buffer.getvalue(), media_type="image/png")
