"""Anomaly detection — flag invoices statistically unusual for their vendor.

Two signals, both relative to the vendor's own history: an *amount spike* (a total far
from the vendor's median, measured with a median/MAD robust z-score so one historical
outlier cannot mask the next) and *threshold gaming* (an amount parked just below a
round approval limit). The detector is pure robust statistics — no ML dependency — and
returns the dominant feature and a human-readable reason with every flag.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from apverify.domain.invoice import Invoice

_ROUND_MANTISSAS = (1, 2, 5)


class AnomalySeverity(Enum):
    NONE = "none"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class AnomalyFeatures:
    amount_robust_z: float
    threshold_proximity: float
    history_size: int


@dataclass(frozen=True, slots=True)
class AnomalyAssessment:
    score: float
    severity: AnomalySeverity
    top_feature: str
    reason: str


def extract_features(
    invoice: Invoice, history: Sequence[Invoice], band: float = 0.05
) -> AnomalyFeatures:
    amount = float(invoice.total.amount)
    amounts = [float(prior.total.amount) for prior in history]
    if not amounts:
        return AnomalyFeatures(0.0, _threshold_proximity(amount, band), 0)
    median = statistics.median(amounts)
    mad = statistics.median([abs(value - median) for value in amounts])
    scale = max(mad, abs(median) * 0.05, 1.0)  # floor: never divide by zero, damp tiny spreads
    return AnomalyFeatures(
        amount_robust_z=abs(amount - median) / scale,
        threshold_proximity=_threshold_proximity(amount, band),
        history_size=len(amounts),
    )


@dataclass(frozen=True, slots=True)
class RobustAnomalyDetector:
    min_history: int = 3
    sensitivity: float = 3.0
    band: float = 0.05
    high: float = 0.8
    medium: float = 0.5
    proximity_z_floor: float = 0.5

    def score(self, invoice: Invoice, history: Sequence[Invoice]) -> AnomalyAssessment:
        features = extract_features(invoice, history, self.band)
        if features.history_size < self.min_history:
            return AnomalyAssessment(0.0, AnomalySeverity.NONE, "history", "insufficient history")
        spike = 1.0 - math.exp(-features.amount_robust_z / self.sensitivity)
        # Gaming = parked near a limit *and* elevated for the vendor; an in-range amount
        # that merely happens to be near a round number is the vendor's normal, not fraud.
        gaming = (
            features.threshold_proximity
            if features.amount_robust_z >= self.proximity_z_floor
            else 0.0
        )
        score = max(spike, gaming)
        top_feature = "amount_spike" if spike >= gaming else "threshold_gaming"
        severity = (
            AnomalySeverity.HIGH
            if score >= self.high
            else AnomalySeverity.MEDIUM
            if score >= self.medium
            else AnomalySeverity.NONE
        )
        return AnomalyAssessment(
            round(score, 4), severity, top_feature, _reason(top_feature, invoice, history, features)
        )


def _threshold_proximity(amount: float, band: float) -> float:
    if amount <= 0:
        return 0.0
    limit = _round_above(amount)
    gap = (limit - amount) / limit
    return 0.0 if gap >= band else 1.0 - gap / band


def _round_above(amount: float) -> float:
    exponent = math.floor(math.log10(amount))
    for mantissa in _ROUND_MANTISSAS:
        candidate = mantissa * 10**exponent
        if candidate > amount:
            return float(candidate)
    return float(10 ** (exponent + 1))


def _reason(
    top_feature: str, invoice: Invoice, history: Sequence[Invoice], features: AnomalyFeatures
) -> str:
    amount = float(invoice.total.amount)
    if top_feature == "threshold_gaming":
        return f"amount {amount:.0f} sits just under the {_round_above(amount):.0f} approval limit"
    median = statistics.median([float(prior.total.amount) for prior in history])
    multiple = amount / median if median else 0.0
    return (
        f"amount {amount:.0f} is {multiple:.1f}x the vendor's median {median:.0f} "
        f"(robust-z {features.amount_robust_z:.1f})"
    )
