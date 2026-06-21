from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tests.support import (
    PO_NUMBER,
    FakeExtractor,
    FakeOcr,
    FakeRenderer,
    build_goods_receipt,
    build_invoice,
    build_purchase_order,
    build_raw_text,
)

from apverify.application.review_payable import ReviewPayableUseCase
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository

_DOC = Path("ignored-by-fakes.pdf")


def _ticking_clock(step: float = 1.0) -> Callable[[], float]:
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += step
        return state["now"]

    return clock


class RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[tuple[str, float]] = []

    def span(self, name: str, detail: str, duration_ms: float) -> None:
        self.spans.append((name, duration_ms))


def _use_case(tracer: RecordingTracer) -> ReviewPayableUseCase:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    return ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(invoice),
        ocr=FakeOcr(build_raw_text(invoice)),
        procurement=InMemoryProcurementRepository(
            purchase_orders=[build_purchase_order()],
            goods_receipts=[build_goods_receipt()],
        ),
        tracer=tracer,
        clock=_ticking_clock(1.0),
    )


def test_each_timed_step_is_measured() -> None:
    review = _use_case(RecordingTracer()).execute(_DOC)
    durations = {entry.step: entry.duration_ms for entry in review.trace}

    # Each timed step opens and closes the clock once (+1.0s steps → 1000 ms).
    assert durations["render"] == 1000.0
    assert durations["extract"] == 1000.0
    assert durations["critic"] == 1000.0
    assert durations["match"] == 1000.0
    assert durations["approve"] == 0.0


def test_tracer_receives_one_span_per_trace_entry() -> None:
    tracer = RecordingTracer()
    review = _use_case(tracer).execute(_DOC)

    assert [name for name, _ in tracer.spans] == [entry.step for entry in review.trace]
