"""Composition root.

The only place adapters are instantiated and wired into the use case. Keeping
construction here means every other module receives its collaborators by
injection and stays free of global state and import-time side effects.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from google import genai
from groq import Groq
from mistralai import Mistral

from apverify.application.observability import NullTracer, Tracer
from apverify.application.ports import (
    AnomalyDetector,
    InvoiceExtractor,
    InvoiceLedger,
    MessageSender,
    ProcurementRepository,
    SelfReportingExtractor,
    SemanticAuditor,
    VendorHistoryRepository,
    VendorMasterRepository,
)
from apverify.application.process_invoice import ProcessInvoiceUseCase
from apverify.application.review_payable import ReviewPayableUseCase
from apverify.domain.anomaly import RobustAnomalyDetector
from apverify.domain.critique import DEFAULT_POLICY, Policy
from apverify.infrastructure.anomaly.history import load_vendor_history
from apverify.infrastructure.fallback import FallbackInvoiceExtractor
from apverify.infrastructure.gemini.extractor import GeminiInvoiceExtractor
from apverify.infrastructure.groq.auditor import GroqSemanticAuditor
from apverify.infrastructure.groq.extractor import GroqInvoiceExtractor
from apverify.infrastructure.invoice_ledger import load_invoice_ledger
from apverify.infrastructure.mistral.extractor import MistralInvoiceExtractor
from apverify.infrastructure.ocr.tesseract import TesseractOcrProvider
from apverify.infrastructure.ollama.extractor import OllamaInvoiceExtractor
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository
from apverify.infrastructure.rendering.pdf import Pdf2ImageRenderer
from apverify.infrastructure.settings import Settings
from apverify.infrastructure.vendor_master.repository import load_vendor_master
from apverify.infrastructure.whatsapp.console import ConsoleMessageSender
from apverify.infrastructure.whatsapp.sender import WhatsAppMessageSender


def build_use_case(
    settings: Settings | None = None, policy: Policy = DEFAULT_POLICY
) -> ProcessInvoiceUseCase:
    settings = settings or Settings()  # values are read from the environment / .env
    return ProcessInvoiceUseCase(
        renderer=Pdf2ImageRenderer(),
        extractor=_build_extractor(settings),
        ocr=TesseractOcrProvider(),
        policy=policy,
    )


def build_review_use_case(
    procurement: ProcurementRepository | None = None,
    settings: Settings | None = None,
    enable_audit: bool = False,
    enable_cross_check: bool = False,
    vendor_master: VendorMasterRepository | None = None,
    anomaly_detector: AnomalyDetector | None = None,
    vendor_history: VendorHistoryRepository | None = None,
    invoice_ledger: InvoiceLedger | None = None,
) -> ReviewPayableUseCase:
    settings = settings or Settings()  # values are read from the environment / .env
    history = vendor_history or _build_vendor_history(settings)
    return ReviewPayableUseCase(
        renderer=Pdf2ImageRenderer(),
        extractor=_build_extractor(settings),
        ocr=TesseractOcrProvider(),
        procurement=procurement or InMemoryProcurementRepository(),
        auditor=_build_auditor(settings) if enable_audit else None,
        secondary_extractor=_build_secondary_extractor(settings) if enable_cross_check else None,
        vendor_master=vendor_master or _build_vendor_master(settings),
        anomaly_detector=anomaly_detector or (RobustAnomalyDetector() if history else None),
        vendor_history=history,
        invoice_ledger=invoice_ledger or _build_invoice_ledger(settings),
        tracer=_build_tracer(settings),
    )


def _build_invoice_ledger(settings: Settings) -> InvoiceLedger | None:
    if not settings.invoice_ledger_path:
        return None
    return load_invoice_ledger(Path(settings.invoice_ledger_path))


def _build_vendor_master(settings: Settings) -> VendorMasterRepository | None:
    if not settings.vendor_master_path:
        return None
    return load_vendor_master(Path(settings.vendor_master_path))


def _build_vendor_history(settings: Settings) -> VendorHistoryRepository | None:
    if not settings.anomaly_history_path:
        return None
    return load_vendor_history(Path(settings.anomaly_history_path))


def build_extractor(settings: Settings | None = None) -> FallbackInvoiceExtractor:
    """The resilient extractor: Gemini, then Groq, then Mistral (then Ollama if enabled)
    — whichever's quota is alive. Supports verbalized confidence (any provider that
    reports it)."""
    return _build_extractor(settings or Settings())


def build_extractor_pair(
    settings: Settings | None = None,
) -> tuple[SelfReportingExtractor, InvoiceExtractor] | None:
    """Two *distinct* providers for cross-model work (fusion): the agreement signal
    is only independent if the second model is a different one. Returns ``None`` when
    fewer than two providers are configured."""
    providers = _build_provider_extractors(settings or Settings())
    if len(providers) < 2:
        return None
    primary, secondary = providers[0], providers[1]
    assert isinstance(primary, SelfReportingExtractor)
    return primary, secondary


def _build_extractor(settings: Settings) -> FallbackInvoiceExtractor:
    return FallbackInvoiceExtractor(_build_provider_extractors(settings))


def build_named_extractors(settings: Settings | None = None) -> dict[str, InvoiceExtractor]:
    """Every configured provider keyed by name, so a caller can route around one
    whose quota is spent — e.g. lead with ``groq`` when ``gemini`` is exhausted."""
    settings = settings or Settings()
    # Gemini leads on quality; Groq (many keys, generous free tier) is preferred over
    # Mistral (burst rate-limited) as the fallback, so evals keep running cheaply.
    # Ollama (local, unlimited) joins only when enabled — the fusion cross-model leg.
    candidates = {
        "gemini": _build_gemini(settings),
        "groq": _build_groq_extractor(settings),
        "mistral": _build_secondary_extractor(settings),
        "ollama": _build_ollama_extractor(settings) if settings.ollama_enabled else None,
    }
    return {name: extractor for name, extractor in candidates.items() if extractor is not None}


def _build_provider_extractors(settings: Settings) -> list[InvoiceExtractor]:
    return list(build_named_extractors(settings).values())


def _build_gemini(settings: Settings) -> GeminiInvoiceExtractor | None:
    keys = settings.gemini_api_keys
    if not keys:
        return None
    clients = [genai.Client(api_key=key) for key in keys]
    return GeminiInvoiceExtractor(clients, settings.gemini_model)


def _build_groq_extractor(settings: Settings) -> GroqInvoiceExtractor | None:
    keys = settings.groq_api_keys
    if not keys:
        return None
    return GroqInvoiceExtractor([Groq(api_key=key) for key in keys], settings.groq_vision_model)


def _build_ollama_extractor(settings: Settings) -> OllamaInvoiceExtractor:
    # Local inference is slow and the model loads cold on first call; allow generous
    # headroom so a real page (or a cold start) does not trip the read timeout.
    client = httpx.Client(base_url=settings.ollama_base_url, timeout=300.0)
    return OllamaInvoiceExtractor(client, settings.ollama_model)


def _build_auditor(settings: Settings) -> SemanticAuditor | None:
    if settings.groq_api_key is None:
        return None
    client = Groq(api_key=settings.groq_api_key.get_secret_value())
    return GroqSemanticAuditor(client, settings.groq_model)


def _build_secondary_extractor(settings: Settings) -> InvoiceExtractor | None:
    keys = settings.mistral_api_keys
    if not keys:
        return None
    return MistralInvoiceExtractor([Mistral(api_key=key) for key in keys], settings.mistral_model)


def build_message_sender(settings: Settings | None = None, dry_run: bool = True) -> MessageSender:
    settings = settings or Settings()  # values are read from the environment / .env
    if dry_run or not settings.whatsapp_enabled:
        return ConsoleMessageSender()
    assert settings.whatsapp_access_token is not None
    return WhatsAppMessageSender(
        access_token=settings.whatsapp_access_token.get_secret_value(),
        phone_number_id=settings.whatsapp_phone_number_id,
        api_version=settings.whatsapp_api_version,
    )


def _build_tracer(settings: Settings) -> Tracer:
    if not settings.langfuse_enabled:
        return NullTracer()
    from langfuse import Langfuse  # lazy: only needed when Langfuse is configured

    from apverify.infrastructure.langfuse.tracer import LangfuseTracer

    assert settings.langfuse_public_key is not None
    assert settings.langfuse_secret_key is not None
    client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    return LangfuseTracer(client)
