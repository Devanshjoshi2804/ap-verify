from __future__ import annotations

from collections.abc import Callable

from apverify.eval.throughput import run_throughput


def _ticking_clock(step: float = 1.0) -> Callable[[], float]:
    state = {"now": 0.0}

    def clock() -> float:
        state["now"] += step
        return state["now"]

    return clock


def test_report_structure_matches_the_batch() -> None:
    report = run_throughput(count=20)

    assert report.invoices == 20
    assert len(report.latencies_ms) == 20
    assert sum(report.decisions.values()) == 20
    assert report.throughput_per_second > 0
    assert report.p95_ms >= report.p50_ms


def test_clean_batch_is_all_auto_approved() -> None:
    report = run_throughput(count=15)
    assert report.decisions == {"AUTO_APPROVE": 15}


def test_corruption_ratio_produces_held_invoices() -> None:
    report = run_throughput(count=12, corrupt_ratio=0.5)

    assert report.decisions.get("AUTO_APPROVE", 0) > 0
    assert sum(count for name, count in report.decisions.items() if name != "AUTO_APPROVE") > 0


def test_latency_and_throughput_are_deterministic_with_an_injected_clock() -> None:
    # Each invoice calls the clock twice (start/end) at +1.0s steps, so every
    # measured latency is exactly 1000 ms; with a wall-clock tick on either side
    # the maths is fully determined.
    report = run_throughput(count=5, clock=_ticking_clock(1.0))

    assert report.latencies_ms == (1000.0,) * 5
    assert report.p50_ms == 1000.0
