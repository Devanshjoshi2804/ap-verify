"""Score anomaly detectors against the labelled benchmark — pure vs Isolation Forest.

The pure robust-statistics detector is always evaluated. If scikit-learn is installed,
the Isolation Forest detector is added for a head-to-head AUROC; if not, the benchmark
reports the pure detector alone and flags that sklearn was unavailable. The ML model
earns its dependency only if it wins.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.application.ports import AnomalyDetector
from apverify.domain.anomaly import AnomalySeverity, RobustAnomalyDetector
from apverify.eval.anomaly_synthesis import AnomalyCase
from apverify.eval.fusion import auroc


@dataclass(frozen=True, slots=True)
class DetectorResult:
    name: str
    auroc: float
    catch_rate: float
    false_positive_rate: float


@dataclass(frozen=True, slots=True)
class AnomalyReport:
    case_count: int
    anomaly_count: int
    results: tuple[DetectorResult, ...]
    sklearn_available: bool


def evaluate_anomaly(cases: Sequence[AnomalyCase]) -> AnomalyReport:
    detectors: list[tuple[str, AnomalyDetector]] = [("robust-statistics", RobustAnomalyDetector())]
    forest = _isolation_forest()
    if forest is not None:
        detectors.append(("isolation-forest", forest))

    return AnomalyReport(
        case_count=len(cases),
        anomaly_count=sum(1 for case in cases if case.is_anomaly),
        results=tuple(_score(name, detector, cases) for name, detector in detectors),
        sklearn_available=forest is not None,
    )


def _score(name: str, detector: AnomalyDetector, cases: Sequence[AnomalyCase]) -> DetectorResult:
    scored = [(detector.score(case.invoice, case.history), case.is_anomaly) for case in cases]
    samples = [(assessment.score, is_anomaly) for assessment, is_anomaly in scored]
    flagged = [
        (assessment.severity is not AnomalySeverity.NONE, is_anomaly)
        for assessment, is_anomaly in scored
    ]
    anomalies = [f for f, is_anomaly in flagged if is_anomaly]
    normals = [f for f, is_anomaly in flagged if not is_anomaly]
    return DetectorResult(
        name=name,
        auroc=auroc(samples),
        catch_rate=_rate([f for f in anomalies if f], anomalies),
        false_positive_rate=_rate([f for f in normals if f], normals),
    )


def _rate(hits: Sequence[object], population: Sequence[object]) -> float:
    return len(hits) / len(population) if population else 0.0


def _isolation_forest() -> AnomalyDetector | None:
    try:
        from apverify.infrastructure.anomaly.isolation_forest import IsolationForestDetector
    except ImportError:
        return None
    return IsolationForestDetector()
