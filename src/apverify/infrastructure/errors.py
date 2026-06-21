"""Adapter-boundary errors.

These wrap failures from external systems (the vision model, the OCR engine, the
PDF rasteriser) in named types so the CLI can report a useful message instead of
leaking a raw SDK traceback.
"""

from __future__ import annotations

from apverify.application.errors import PortError


class AdapterError(PortError):
    """Base class for failures originating in an infrastructure adapter."""


class ExtractionError(AdapterError):
    """The vision model failed to return a usable invoice."""


class OcrError(AdapterError):
    """The OCR engine could not read the page."""


class RenderError(AdapterError):
    """The document could not be rasterised into page images."""


class ProcurementError(AdapterError):
    """Procurement master data could not be loaded."""


class AuditError(AdapterError):
    """The semantic auditor failed to return a usable judgement."""


class MessagingError(AdapterError):
    """A message could not be delivered."""
