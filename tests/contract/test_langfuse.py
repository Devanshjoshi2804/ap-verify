"""Live Langfuse adapter test.

Skipped unless LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are set (and the
`langfuse` extra is installed). Verifies a span can be emitted and flushed to a
real Langfuse instance without error.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.contract

_KEYS_PRESENT = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@pytest.mark.skipif(not _KEYS_PRESENT, reason="no LANGFUSE_* keys in environment")
def test_langfuse_tracer_emits_and_flushes() -> None:
    from langfuse import Langfuse

    from apverify.infrastructure.langfuse.tracer import LangfuseTracer

    client = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    tracer = LangfuseTracer(client)
    tracer.span("critic", "AUTO_APPROVE @ 100%", 5.9)
    client.flush()
