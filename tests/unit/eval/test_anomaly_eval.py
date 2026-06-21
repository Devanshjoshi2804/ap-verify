from __future__ import annotations

from apverify.eval.anomaly_eval import evaluate_anomaly
from apverify.eval.anomaly_synthesis import build_anomaly_cases
from apverify.eval.synthetic import generate_dataset


def test_pure_detector_separates_anomalies_with_no_false_positives() -> None:
    # A larger n exercises vendors whose normal amount lands near a round number; the
    # z-floor gate keeps those out of the threshold-gaming flag, so FP stays at zero.
    report = evaluate_anomaly(build_anomaly_cases(generate_dataset(count=25)))
    pure = next(r for r in report.results if r.name == "robust-statistics")
    assert pure.catch_rate >= 0.9
    assert pure.false_positive_rate == 0.0
    assert pure.auroc >= 0.9


def test_report_always_includes_the_pure_detector() -> None:
    report = evaluate_anomaly(build_anomaly_cases(generate_dataset(count=5)))
    assert any(r.name == "robust-statistics" for r in report.results)


def test_empty_cases_yield_a_zeroed_report() -> None:
    report = evaluate_anomaly([])
    assert report.case_count == 0
