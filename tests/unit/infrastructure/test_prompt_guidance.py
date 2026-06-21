"""Every provider's extraction prompt must carry the same vendor guidance.

The prompts are defined per provider (their JSON conventions differ), so the
seller-vs-buyer instruction is a single shared constant that each one embeds —
this guards against a provider drifting and silently dropping it.
"""

from __future__ import annotations

from apverify.infrastructure.gemini.extractor import _PROMPT as GEMINI_PROMPT
from apverify.infrastructure.groq.extractor import _PROMPT as GROQ_PROMPT
from apverify.infrastructure.mapping import LINE_ITEM_GUIDANCE, VENDOR_GUIDANCE
from apverify.infrastructure.mistral.extractor import _PROMPT as MISTRAL_PROMPT
from apverify.infrastructure.ollama.extractor import _PROMPT as OLLAMA_PROMPT

_PROMPTS = (GEMINI_PROMPT, GROQ_PROMPT, OLLAMA_PROMPT, MISTRAL_PROMPT)


def test_vendor_guidance_names_the_seller_not_the_buyer() -> None:
    lowered = VENDOR_GUIDANCE.lower()
    assert "seller" in lowered
    assert "buyer" in lowered


def test_line_item_guidance_forbids_merging_repeated_rows() -> None:
    lowered = LINE_ITEM_GUIDANCE.lower()
    assert "row" in lowered
    assert "merge" in lowered or "separate" in lowered


def test_every_provider_prompt_embeds_the_vendor_guidance() -> None:
    for prompt in _PROMPTS:
        assert VENDOR_GUIDANCE in prompt


def test_every_provider_prompt_embeds_the_line_item_guidance() -> None:
    for prompt in _PROMPTS:
        assert LINE_ITEM_GUIDANCE in prompt
