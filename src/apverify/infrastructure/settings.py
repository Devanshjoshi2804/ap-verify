"""Typed application settings — the single place environment variables are read.

Everything else receives configuration through constructor arguments, so no
business code reaches into ``os.environ`` and there is exactly one boundary to
audit for secret handling.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Optional: the highest-quality extractor. Absent in a local-first (Ollama-only)
    # deployment, so the app must still start with no Gemini key.
    gemini_api_key: SecretStr | None = Field(None, alias="GEMINI_API_KEY")
    gemini_fallback_keys: str = Field("", alias="GEMINI_FALLBACK_KEYS")
    gemini_model: str = Field("gemini-flash-latest", alias="GEMINI_MODEL")

    # Optional: enables the LLM-as-auditor critic layer when present.
    groq_api_key: SecretStr | None = Field(None, alias="GROQ_API_KEY")
    groq_fallback_keys: str = Field("", alias="GROQ_FALLBACK_KEYS")
    groq_model: str = Field("llama-3.3-70b-versatile", alias="GROQ_MODEL")
    # Vision model backing the Groq fallback extractor (distinct from the text auditor).
    groq_vision_model: str = Field(
        "meta-llama/llama-4-scout-17b-16e-instruct", alias="GROQ_VISION_MODEL"
    )

    # Optional: enables the second-model self-consistency check when present.
    mistral_api_key: SecretStr | None = Field(None, alias="MISTRAL_API_KEY")
    mistral_fallback_keys: str = Field("", alias="MISTRAL_FALLBACK_KEYS")
    mistral_model: str = Field("pixtral-12b-2409", alias="MISTRAL_MODEL")

    # Optional: a JSON vendor master enabling bank-change / impersonation (BEC) checks.
    vendor_master_path: str = Field("", alias="VENDOR_MASTER_PATH")

    # Optional: a JSON of prior invoices enabling per-vendor anomaly detection.
    anomaly_history_path: str = Field("", alias="ANOMALY_HISTORY_PATH")

    # Optional: a JSON ledger of posted invoices enabling duplicate-payment detection.
    invoice_ledger_path: str = Field("", alias="INVOICE_LEDGER_PATH")

    # Optional: a local, unlimited extractor — the independent cross-model leg for fusion.
    ollama_enabled: bool = Field(False, alias="OLLAMA_ENABLED")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("qwen2.5vl:7b", alias="OLLAMA_MODEL")

    # Optional: ships pipeline traces to Langfuse when both keys are present.
    langfuse_public_key: SecretStr | None = Field(None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    @property
    def langfuse_enabled(self) -> bool:
        return self.langfuse_public_key is not None and self.langfuse_secret_key is not None

    # Optional: WhatsApp Cloud API for the AR-collections agent + inbound webhook.
    whatsapp_access_token: SecretStr | None = Field(None, alias="WHATSAPP_ACCESS_TOKEN")
    whatsapp_phone_number_id: str = Field("", alias="WHATSAPP_PHONE_NUMBER_ID")
    whatsapp_app_secret: SecretStr | None = Field(None, alias="WHATSAPP_APP_SECRET")
    whatsapp_verify_token: str = Field("", alias="WHATSAPP_VERIFY_TOKEN")
    whatsapp_api_version: str = Field("v21.0", alias="WHATSAPP_API_VERSION")

    @property
    def whatsapp_enabled(self) -> bool:
        return self.whatsapp_access_token is not None and bool(self.whatsapp_phone_number_id)

    @property
    def gemini_api_keys(self) -> list[str]:
        """The primary key first, then any fallbacks tried on rate-limit; empty in a
        local-first deployment with no Gemini key configured."""
        primary = [self.gemini_api_key.get_secret_value()] if self.gemini_api_key else []
        return [*primary, *_split(self.gemini_fallback_keys)]

    @property
    def mistral_api_keys(self) -> list[str]:
        primary = [self.mistral_api_key.get_secret_value()] if self.mistral_api_key else []
        return [*primary, *_split(self.mistral_fallback_keys)]

    @property
    def groq_api_keys(self) -> list[str]:
        primary = [self.groq_api_key.get_secret_value()] if self.groq_api_key else []
        return [*primary, *_split(self.groq_fallback_keys)]


def _split(value: str) -> list[str]:
    return [key.strip() for key in value.split(",") if key.strip()]
