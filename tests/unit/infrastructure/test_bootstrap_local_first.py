"""Local-first wiring: the pipeline must build with no cloud API keys, using only
the local Ollama extractor."""

from __future__ import annotations

from apverify.infrastructure.settings import Settings
from apverify.interface.cli.bootstrap import build_named_extractors


def _keyless(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_named_extractors_is_ollama_only_when_only_ollama_is_configured() -> None:
    extractors = build_named_extractors(_keyless(OLLAMA_ENABLED=True))
    assert list(extractors) == ["ollama"]


def test_named_extractors_is_empty_with_no_providers_at_all() -> None:
    # No keys and Ollama disabled: an empty registry rather than a crash on Gemini.
    assert build_named_extractors(_keyless()) == {}
