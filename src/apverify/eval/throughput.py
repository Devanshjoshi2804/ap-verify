"""Throughput harness — how fast does the verification layer run, and at what mix?

The critic is the part that runs on every invoice forever, so its cost is the one
that compounds at scale. This measures it directly: latency per invoice,
sustained throughput, and the resulting approve/hold/review mix, over a synthetic
batch with an optional injected-error fraction.

The clock is injected so the latency and throughput maths are deterministic under
test; in production the real bottleneck is the I/O-bound extraction call, which
the worker pool is sized for.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from apverify.domain.checks import review
from apverify.domain.critique import DEFAULT_POLICY, ApprovalDecision, Policy
from apverify.domain.invoice import Invoice
from apverify.domain.ocr import RawText
from apverify.eval.corruptor import corruptions
from apverify.eval.synthetic import faithful_raw_text, generate_dataset

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class ThroughputReport:
    invoices: int
    workers: int
    wall_seconds: float
    latencies_ms: tuple[float, ...]
    decisions: dict[str, int]

    @property
    def throughput_per_second(self) -> float:
        return self.invoices / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def projected_per_day(self) -> int:
        return int(self.throughput_per_second * 86_400)

    @property
    def mean_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p50_ms(self) -> float:
        return _percentile(self.latencies_ms, 0.50)

    @property
    def p95_ms(self) -> float:
        return _percentile(self.latencies_ms, 0.95)

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms, default=0.0)


def run_throughput(
    count: int = 500,
    workers: int = 1,
    corrupt_ratio: float = 0.0,
    policy: Policy = DEFAULT_POLICY,
    clock: Clock = time.perf_counter,
) -> ThroughputReport:
    items = _prepare(count, corrupt_ratio)

    def measure(item: tuple[Invoice, RawText]) -> tuple[float, ApprovalDecision]:
        invoice, raw_text = item
        started = clock()
        decision = review(invoice, raw_text, policy).decision
        return (clock() - started) * 1000.0, decision

    wall_start = clock()
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(measure, items))
    else:
        results = [measure(item) for item in items]
    wall_seconds = clock() - wall_start

    decisions = Counter(decision.value for _, decision in results)
    return ThroughputReport(
        invoices=len(results),
        workers=workers,
        wall_seconds=wall_seconds,
        latencies_ms=tuple(latency for latency, _ in results),
        decisions=dict(decisions),
    )


def _prepare(count: int, corrupt_ratio: float) -> list[tuple[Invoice, RawText]]:
    kinds = corruptions()
    stride = round(1 / corrupt_ratio) if corrupt_ratio > 0 else 0
    items: list[tuple[Invoice, RawText]] = []
    for index, ground_truth in enumerate(generate_dataset(count)):
        raw_text = faithful_raw_text(ground_truth.invoice)
        invoice = ground_truth.invoice
        if stride and index % stride == 0:
            invoice = kinds[index % len(kinds)].apply(invoice)
        items.append((invoice, raw_text))
    return items


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
