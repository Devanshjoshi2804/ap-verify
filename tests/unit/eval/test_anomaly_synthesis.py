from __future__ import annotations

from apverify.eval.anomaly_synthesis import ANOMALY_KINDS, SCENARIOS, build_anomaly_cases
from apverify.eval.synthetic import generate_dataset


def test_each_base_invoice_yields_one_case_per_scenario() -> None:
    cases = build_anomaly_cases(generate_dataset(count=3))
    assert {case.kind for case in cases} == set(SCENARIOS)
    assert all(len(case.history) >= 3 for case in cases)  # enough history to score


def test_anomaly_kinds_are_labelled_anomalous_and_normal_is_not() -> None:
    for case in build_anomaly_cases(generate_dataset(count=3)):
        assert case.is_anomaly == (case.kind in ANOMALY_KINDS)
