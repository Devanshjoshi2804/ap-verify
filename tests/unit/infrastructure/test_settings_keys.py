from __future__ import annotations

from apverify.infrastructure.settings import Settings


def _settings(**overrides: str) -> Settings:
    base = {"GEMINI_API_KEY": "primary"}
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[arg-type]


def test_groq_keys_put_the_primary_first_then_fallbacks() -> None:
    settings = _settings(GROQ_API_KEY="g0", GROQ_FALLBACK_KEYS="g1, g2 ,g3")
    assert settings.groq_api_keys == ["g0", "g1", "g2", "g3"]


def test_groq_keys_are_empty_without_a_primary() -> None:
    assert _settings().groq_api_keys == []


def test_settings_construct_without_any_gemini_key() -> None:
    # Local-first: the app must start with no cloud keys at all (Ollama only).
    settings = Settings(_env_file=None)
    assert settings.gemini_api_keys == []


def test_gemini_keys_put_the_primary_first_then_fallbacks() -> None:
    settings = _settings(GEMINI_FALLBACK_KEYS="k1, k2")
    assert settings.gemini_api_keys == ["primary", "k1", "k2"]
