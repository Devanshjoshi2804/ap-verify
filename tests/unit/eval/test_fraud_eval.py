from __future__ import annotations

from apverify.eval.fraud_eval import evaluate_fraud
from apverify.eval.fraud_synthesis import build_fraud_cases
from apverify.eval.synthetic import generate_dataset


def test_exact_and_ocr_variants_are_caught_with_no_false_positives() -> None:
    report = evaluate_fraud(build_fraud_cases(generate_dataset(count=6)))
    # The unambiguous duplicate types are fully caught and no legit case is flagged.
    assert report.per_kind["exact_resend"] == 1.0
    assert report.per_kind["ocr_variant"] == 1.0
    assert report.false_positive_rate == 0.0


def test_recurring_retainer_is_never_flagged() -> None:
    report = evaluate_fraud(build_fraud_cases(generate_dataset(count=6)))
    assert report.per_kind["legit_recurring"] == 0.0


def test_score_separates_fraud_from_legitimate() -> None:
    report = evaluate_fraud(build_fraud_cases(generate_dataset(count=6)))
    assert report.auroc >= 0.9


def test_empty_cases_yield_a_zeroed_report() -> None:
    report = evaluate_fraud([])
    assert report.case_count == 0
    assert report.catch_rate == 0.0
