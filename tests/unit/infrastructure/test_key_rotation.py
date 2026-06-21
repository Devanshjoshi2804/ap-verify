from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors as genai_errors
from groq import APIError as GroqAPIError

from apverify.application.ports import PageImage
from apverify.domain.critique import InvoiceField
from apverify.infrastructure.errors import ExtractionError
from apverify.infrastructure.gemini.extractor import GeminiInvoiceExtractor
from apverify.infrastructure.groq.extractor import GroqInvoiceExtractor
from apverify.infrastructure.mapping import ConfidentInvoiceDTO, FieldConfidenceDTO, InvoiceDTO
from apverify.infrastructure.mistral.extractor import MistralInvoiceExtractor

_PAGES = [PageImage(data=b"image-bytes")]
_DTO = InvoiceDTO(
    vendor_name="ACME Steel Pvt Ltd",
    invoice_number="INV-1",
    invoice_date="04-06-2025",
    subtotal="100",
    tax="18",
    total="118",
)
_VALID_JSON = _DTO.model_dump_json()


def _gemini_client(behaviour: Any) -> Any:
    return SimpleNamespace(models=SimpleNamespace(generate_content=behaviour))


def _quota_exceeded(**_: Any) -> Any:
    raise genai_errors.APIError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})


def _returns_dto(**_: Any) -> Any:
    return SimpleNamespace(parsed=_DTO)


def test_gemini_rotates_to_the_next_key_on_rate_limit() -> None:
    extractor = GeminiInvoiceExtractor(
        [_gemini_client(_quota_exceeded), _gemini_client(_returns_dto)], "gemini-flash-latest"
    )
    assert extractor.extract(_PAGES).vendor_name == "ACME Steel Pvt Ltd"


def test_gemini_raises_when_every_key_is_exhausted() -> None:
    extractor = GeminiInvoiceExtractor(
        [_gemini_client(_quota_exceeded), _gemini_client(_quota_exceeded)], "gemini-flash-latest"
    )
    with pytest.raises(ExtractionError, match="across 2 key"):
        extractor.extract(_PAGES)


def _confident_dto() -> ConfidentInvoiceDTO:
    return ConfidentInvoiceDTO(
        **_DTO.model_dump(),
        field_confidence=FieldConfidenceDTO(vendor_name=0.95, total=0.7),
    )


def test_gemini_extracts_verbalized_confidence() -> None:
    dto = _confident_dto()
    extractor = GeminiInvoiceExtractor(
        [_gemini_client(lambda **_: SimpleNamespace(parsed=dto))], "gemini-flash-latest"
    )

    extraction = extractor.extract_with_confidence(_PAGES)

    assert extraction.invoice.vendor_name == "ACME Steel Pvt Ltd"
    assert extraction.confidences[InvoiceField.VENDOR] == 0.95
    assert extraction.confidences[InvoiceField.TOTAL] == 0.7


def test_gemini_requires_at_least_one_client() -> None:
    with pytest.raises(ValueError, match="at least one"):
        GeminiInvoiceExtractor([], "gemini-flash-latest")


def _mistral_client(content: str) -> Any:
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(complete=lambda **_: response))


def test_mistral_rotates_past_unusable_output() -> None:
    extractor = MistralInvoiceExtractor(
        [_mistral_client("{ not valid json"), _mistral_client(_VALID_JSON)], "pixtral-12b-2409"
    )
    assert str(extractor.extract(_PAGES).total) == "118.00"


_CONFIDENT_JSON = _confident_dto().model_dump_json()


def test_mistral_extracts_verbalized_confidence() -> None:
    extractor = MistralInvoiceExtractor([_mistral_client(_CONFIDENT_JSON)], "pixtral-12b-2409")
    result = extractor.extract_with_confidence(_PAGES)
    assert result.confidences[InvoiceField.VENDOR] == 0.95


def _groq_client(behaviour: Any) -> Any:
    completions = SimpleNamespace(create=behaviour)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _groq_returns(content: str) -> Any:
    def _create(**_: Any) -> Any:
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return _create


def _groq_rate_limited(**_: Any) -> Any:
    raise GroqAPIError("429", request=httpx.Request("POST", "http://groq"), body=None)


def test_groq_rotates_to_the_next_key_on_failure() -> None:
    extractor = GroqInvoiceExtractor(
        [_groq_client(_groq_rate_limited), _groq_client(_groq_returns(_VALID_JSON))], "scout"
    )
    assert str(extractor.extract(_PAGES).total) == "118.00"


def test_groq_extracts_verbalized_confidence() -> None:
    extractor = GroqInvoiceExtractor([_groq_client(_groq_returns(_CONFIDENT_JSON))], "scout")
    result = extractor.extract_with_confidence(_PAGES)
    assert result.confidences[InvoiceField.VENDOR] == 0.95
