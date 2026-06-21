"""Score the three fraud detectors together over the unified suite.

Every case is run through all three detectors at the live-pipeline thresholds; a case is
flagged if any fires. The combined catch-rate and the system-wide false-positive rate
(a clean invoice flagged by *any* detector — cross-talk) are the capstone numbers, with
a per-label catch table and per-detector attribution beneath them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.domain.anomaly import AnomalySeverity, RobustAnomalyDetector
from apverify.domain.fraud import find_duplicates
from apverify.domain.vendor_master import Severity, assess_vendor_risk
from apverify.eval.fraud_suite_synthesis import SuiteCase

_DETECTORS = ("duplicate", "bec", "anomaly")


@dataclass(frozen=True, slots=True)
class FraudSuiteReport:
    case_count: int
    fraud_count: int
    catch_rate: float
    false_positive_rate: float
    precision: float
    per_label: dict[str, float]
    per_detector: dict[str, int]


def evaluate_fraud_suite(cases: Sequence[SuiteCase]) -> FraudSuiteReport:
    anomaly = RobustAnomalyDetector()
    scored = [(case, _fired(case, anomaly)) for case in cases]

    frauds = [(case, fired) for case, fired in scored if case.is_fraud]
    cleans = [(case, fired) for case, fired in scored if not case.is_fraud]
    flagged = [(case, fired) for case, fired in scored if fired]

    return FraudSuiteReport(
        case_count=len(cases),
        fraud_count=len(frauds),
        catch_rate=_rate([1 for _, fired in frauds if fired], frauds),
        false_positive_rate=_rate([1 for _, fired in cleans if fired], cleans),
        precision=_rate([1 for case, _ in flagged if case.is_fraud], flagged),
        per_label=_per_label(scored),
        per_detector=_per_detector(scored),
    )


def _fired(case: SuiteCase, anomaly: RobustAnomalyDetector) -> frozenset[str]:
    fired: set[str] = set()
    if find_duplicates(case.invoice, case.priors):
        fired.add("duplicate")
    if assess_vendor_risk(case.invoice, case.master).severity is Severity.HIGH:
        fired.add("bec")
    if anomaly.score(case.invoice, case.history).severity in (
        AnomalySeverity.HIGH,
        AnomalySeverity.MEDIUM,
    ):
        fired.add("anomaly")
    return frozenset(fired)


def _rate(hits: Sequence[object], population: Sequence[object]) -> float:
    return len(hits) / len(population) if population else 0.0


def _per_label(scored: Sequence[tuple[SuiteCase, frozenset[str]]]) -> dict[str, float]:
    labels = sorted({case.label for case, _ in scored})
    return {
        label: _rate(
            [1 for case, fired in scored if case.label == label and fired],
            [1 for case, _ in scored if case.label == label],
        )
        for label in labels
    }


def _per_detector(scored: Sequence[tuple[SuiteCase, frozenset[str]]]) -> dict[str, int]:
    return {
        detector: sum(1 for case, fired in scored if case.is_fraud and detector in fired)
        for detector in _DETECTORS
    }
