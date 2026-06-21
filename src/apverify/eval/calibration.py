"""Confidence calibration — does a stated confidence mean what it says?

A critic that auto-approves at "90% confidence" is only trustworthy if invoices it
rates 0.9 are right ~90% of the time. This module measures that, purely, over
``(confidence, correct)`` samples:

* **Expected Calibration Error (ECE)** — average gap between confidence and
  accuracy across confidence bins. Lower is better; 0 is perfectly calibrated.
* **Reliability bins** — the data behind a reliability diagram.
* **Risk-coverage** — at each confidence threshold, how much is auto-approved
  (coverage) and how often that's wrong (error). The operating point is the
  threshold with the most coverage at **zero wrong auto-approvals** — the formal
  version of the safe-auto-approval metric.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

Sample = tuple[float, bool]

_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    confidence: float  # mean confidence of samples in the bin
    accuracy: float  # fraction correct in the bin
    count: int


@dataclass(frozen=True, slots=True)
class CoveragePoint:
    threshold: float
    coverage: float  # fraction of items at or above the threshold
    error: float  # fraction of covered items that are wrong


def reliability_bins(samples: Sequence[Sample], bins: int = 10) -> list[ReliabilityBin]:
    edges = [i / bins for i in range(bins + 1)]
    result: list[ReliabilityBin] = []
    for lower, upper in pairwise(edges):
        in_bin = [(c, ok) for c, ok in samples if lower <= c < upper or (upper == 1.0 and c == 1.0)]
        if not in_bin:
            continue
        count = len(in_bin)
        result.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                confidence=sum(c for c, _ in in_bin) / count,
                accuracy=sum(1 for _, ok in in_bin if ok) / count,
                count=count,
            )
        )
    return result


def expected_calibration_error(samples: Sequence[Sample], bins: int = 10) -> float:
    total = len(samples)
    if not total:
        return 0.0
    return sum(
        (b.count / total) * abs(b.confidence - b.accuracy) for b in reliability_bins(samples, bins)
    )


def risk_coverage(samples: Sequence[Sample], steps: int = 10) -> list[CoveragePoint]:
    total = len(samples)
    if not total:
        return []
    points: list[CoveragePoint] = []
    for step in range(steps + 1):
        threshold = step / steps
        covered = [ok for c, ok in samples if c >= threshold]
        coverage = len(covered) / total
        error = sum(1 for ok in covered if not ok) / len(covered) if covered else 0.0
        points.append(CoveragePoint(threshold=threshold, coverage=coverage, error=error))
    return points


def apply_temperature(confidence: float, temperature: float) -> float:
    """Re-scale a confidence by temperature ``T``: ``sigmoid(logit(p) / T)``.

    ``T > 1`` softens overconfident scores toward 0.5; ``T = 1`` is the identity.
    """
    clamped = min(max(confidence, _EPS), 1 - _EPS)
    logit = math.log(clamped / (1 - clamped))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def fit_temperature(samples: Sequence[Sample]) -> float:
    """The temperature minimising negative log-likelihood on the samples.

    A scan over a log-spaced grid — no optimiser dependency, deterministic, and
    plenty precise for a one-parameter fit.
    """
    if not samples:
        return 1.0
    # 0.1 .. 100: the wide upper end matters for cheap, badly-overconfident extractors
    # (a small free-tier model can need T > 10 before its scores stop lying).
    candidates = [10 ** (-1 + 3 * i / 200) for i in range(201)]
    return min(candidates, key=lambda t: _nll(samples, t))


def temperature_scaled(samples: Sequence[Sample], temperature: float) -> list[Sample]:
    return [
        (apply_temperature(confidence, temperature), correct) for confidence, correct in samples
    ]


def _nll(samples: Sequence[Sample], temperature: float) -> float:
    total = 0.0
    for confidence, correct in samples:
        probability = apply_temperature(confidence, temperature)
        probability = min(max(probability, _EPS), 1 - _EPS)
        total -= math.log(probability) if correct else math.log(1 - probability)
    return total / len(samples)


def operating_point_at(
    samples: Sequence[Sample], max_error: float = 0.0, steps: int = 20
) -> CoveragePoint:
    """The threshold with the most coverage whose error stays within ``max_error`` — the
    selective-autonomy operating point for a given error budget. When even the budget
    cannot be met, the lowest-error point (ties broken toward more coverage).

    ``max_error=0.0`` is the strict "auto-approve only what is provably error-free"
    case; a small budget (say 0.02) reports how much *more* can be auto-approved if a
    little error is tolerated — the realistic autonomy curve."""
    covered = [p for p in risk_coverage(samples, steps) if p.coverage > 0.0]
    if not covered:
        return CoveragePoint(1.0, 0.0, 0.0)
    within_budget = [p for p in covered if p.error <= max_error]
    if within_budget:
        return max(within_budget, key=lambda p: p.coverage)
    return min(covered, key=lambda p: (p.error, -p.coverage))


def best_operating_point(samples: Sequence[Sample], steps: int = 20) -> CoveragePoint:
    """The threshold with the most coverage at zero error; if none is error-free,
    the lowest-error point (ties broken toward more coverage)."""
    return operating_point_at(samples, max_error=0.0, steps=steps)
