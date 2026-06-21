"""Unified, structured explanations behind a fraud decision.

Every detector emits a free-text reason; this turns each decision into a *ranked* set of
contributing factors that read the same way across sources. The rule detectors are
glass-box, so their factors enumerate the conditions that fired; the fusion model's
factors (built in ``eval``) are exact ``weight * feature`` linear contributions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.fraud import DuplicateMatch
from apverify.domain.vendor_master import Severity, VendorRiskAssessment

_SEVERITY_WEIGHT = {
    Severity.HIGH: 1.0,
    Severity.LOW: 0.3,
    Severity.NONE: 0.0,
    AnomalySeverity.HIGH: 1.0,
    AnomalySeverity.MEDIUM: 0.6,
    AnomalySeverity.NONE: 0.0,
}


@dataclass(frozen=True, slots=True)
class Factor:
    signal: str
    value: str
    contribution: float  # signed; positive = toward the flag (toward P(correct) for fusion)
    detail: str


@dataclass(frozen=True, slots=True)
class Explanation:
    source: str
    headline: str
    factors: tuple[Factor, ...]


def explanation(source: str, headline: str, factors: Sequence[Factor]) -> Explanation:
    ranked = tuple(sorted(factors, key=lambda factor: abs(factor.contribution), reverse=True))
    return Explanation(source=source, headline=headline, factors=ranked)


def explain_duplicate(match: DuplicateMatch) -> Explanation:
    return explanation(
        "duplicate",
        f"{match.tier.value} (score {match.score:.2f})",
        [
            Factor("duplicate_tier", match.tier.value, match.score, match.reason),
            Factor(
                "matched_id", match.matched_id, match.score * 0.5, f"matched {match.matched_id}"
            ),
        ],
    )


def explain_vendor_risk(assessment: VendorRiskAssessment) -> Explanation:
    return explanation(
        "vendor-master",
        f"{assessment.kind.value} ({assessment.severity.value})",
        [
            Factor(
                "vendor_risk",
                assessment.kind.value,
                _SEVERITY_WEIGHT[assessment.severity],
                assessment.reason,
            ),
            Factor(
                "name_similarity",
                f"{assessment.score:.2f}",
                assessment.score * 0.5,
                f"nearest known vendor: {assessment.matched_vendor}",
            ),
        ],
    )


def explain_anomaly(assessment: AnomalyAssessment) -> Explanation:
    return explanation(
        "anomaly",
        f"{assessment.top_feature} ({assessment.severity.value})",
        [
            Factor(
                assessment.top_feature,
                f"{assessment.score:.2f}",
                assessment.score,
                assessment.reason,
            )
        ],
    )
