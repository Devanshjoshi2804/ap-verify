"""Langfuse adapter for the ``Tracer`` port.

Ships one event per pipeline step to Langfuse (name + outcome + duration), giving
production traces with timing per invoice. The adapter depends on a minimal client
Protocol rather than importing the SDK, so the domain stays SDK-free and the unit
test runs without the optional ``langfuse`` extra installed; the concrete
``langfuse.Langfuse`` client is constructed at the composition root.
"""

from __future__ import annotations

from typing import Any, Protocol


class LangfuseClient(Protocol):
    def create_event(self, *, name: str, metadata: Any) -> Any: ...


class LangfuseTracer:
    def __init__(self, client: LangfuseClient) -> None:
        self._client = client

    def span(self, name: str, detail: str, duration_ms: float) -> None:
        self._client.create_event(
            name=name, metadata={"detail": detail, "duration_ms": duration_ms}
        )
