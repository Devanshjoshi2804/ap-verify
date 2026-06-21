from __future__ import annotations

from typing import Any

from apverify.infrastructure.langfuse.tracer import LangfuseTracer


class _RecordingClient:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def create_event(self, *, name: str, metadata: Any) -> Any:
        self.events.append({"name": name, "metadata": metadata})
        return None


def test_span_emits_a_langfuse_event_with_timing() -> None:
    client = _RecordingClient()
    tracer = LangfuseTracer(client)

    tracer.span("extract", "ACME · total 184200.00", 21351.5)

    assert client.events == [
        {
            "name": "extract",
            "metadata": {"detail": "ACME · total 184200.00", "duration_ms": 21351.5},
        }
    ]
