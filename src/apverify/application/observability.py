"""Tracing seam for the pipeline.

The orchestrator emits one span per step (name, outcome, wall-clock duration) to a
``Tracer``. The default discards them — the timing already lives on the returned
trace — but a production deployment injects an adapter that ships spans to an
observability backend (Langfuse, OTel) without the use case knowing or caring.
"""

from __future__ import annotations

from typing import Protocol


class Tracer(Protocol):
    def span(self, name: str, detail: str, duration_ms: float) -> None:
        """Record one completed pipeline step."""
        ...


class NullTracer:
    """Discards spans; the default when no backend is configured."""

    def span(self, name: str, detail: str, duration_ms: float) -> None:
        return None
