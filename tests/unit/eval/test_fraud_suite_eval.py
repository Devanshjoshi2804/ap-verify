from __future__ import annotations

from apverify.eval.fraud_suite_eval import evaluate_fraud_suite
from apverify.eval.fraud_suite_synthesis import build_fraud_suite
from apverify.eval.synthetic import generate_dataset


def test_every_fraud_is_caught_with_no_false_positives() -> None:
    report = evaluate_fraud_suite(build_fraud_suite(generate_dataset(count=5)))
    assert report.catch_rate == 1.0
    assert report.false_positive_rate == 0.0


def test_each_fraud_type_is_attributed_to_its_own_detector() -> None:
    report = evaluate_fraud_suite(build_fraud_suite(generate_dataset(count=5)))
    assert report.per_label["dup_resend"] == 1.0
    assert report.per_label["bank_change"] == 1.0
    assert report.per_label["amount_spike"] == 1.0
    assert report.per_label["clean"] == 0.0
    assert report.per_detector["duplicate"] >= 1
    assert report.per_detector["bec"] >= 1
    assert report.per_detector["anomaly"] >= 1


def test_empty_cases_yield_a_zeroed_report() -> None:
    report = evaluate_fraud_suite([])
    assert report.case_count == 0
    assert report.catch_rate == 0.0
