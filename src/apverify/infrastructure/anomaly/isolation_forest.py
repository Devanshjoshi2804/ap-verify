"""Isolation Forest anomaly detector — the optional ML challenger to the pure baseline.

Fits an Isolation Forest on the vendor's historical totals and scores the candidate's
total against it. It sees only the amount (not the threshold-proximity feature the pure
detector engineers), so the benchmark shows where domain knowledge beats a generic ML
model. Lives behind the optional ``anomaly`` extra and is used only by the eval harness.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from sklearn.ensemble import IsolationForest

from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.invoice import Invoice


class IsolationForestDetector:
    def __init__(self, min_history: int = 3, high: float = 0.8, medium: float = 0.5) -> None:
        self._min_history = min_history
        self._high = high
        self._medium = medium

    def score(self, invoice: Invoice, history: Sequence[Invoice]) -> AnomalyAssessment:
        amounts = [[float(prior.total.amount)] for prior in history]
        if len(amounts) < self._min_history:
            return AnomalyAssessment(0.0, AnomalySeverity.NONE, "history", "insufficient history")
        model = IsolationForest(random_state=0, n_estimators=100).fit(amounts)
        decision = float(model.decision_function([[float(invoice.total.amount)]])[0])
        score = 1.0 / (1.0 + math.exp(decision))  # lower decision => more anomalous => higher score
        severity = (
            AnomalySeverity.HIGH
            if score >= self._high
            else AnomalySeverity.MEDIUM
            if score >= self._medium
            else AnomalySeverity.NONE
        )
        return AnomalyAssessment(
            round(score, 4), severity, "isolation_forest", "isolation-forest amount outlier score"
        )
