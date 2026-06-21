from __future__ import annotations

from apverify.eval.bec_eval import evaluate_bec
from apverify.eval.bec_synthesis import build_bec_cases
from apverify.eval.synthetic import generate_dataset


def test_bank_change_and_impersonation_are_caught_with_no_false_positives() -> None:
    report = evaluate_bec(build_bec_cases(generate_dataset(count=5)))
    assert report.per_kind["bank_change"] == 1.0
    assert report.per_kind["impersonation"] == 1.0
    assert report.false_positive_rate == 0.0


def test_new_payee_is_never_flagged_high() -> None:
    report = evaluate_bec(build_bec_cases(generate_dataset(count=5)))
    assert report.per_kind["new_payee"] == 0.0
    assert report.per_kind["legit_new"] == 0.0


def test_impersonation_score_separates_from_legitimate_new_vendors() -> None:
    report = evaluate_bec(build_bec_cases(generate_dataset(count=5)))
    assert report.impersonation_auroc >= 0.9


def test_empty_cases_yield_a_zeroed_report() -> None:
    report = evaluate_bec([])
    assert report.case_count == 0
    assert report.catch_rate == 0.0
