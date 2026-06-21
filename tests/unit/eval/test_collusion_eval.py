from __future__ import annotations

from apverify.eval.collusion_eval import evaluate_collusion
from apverify.eval.collusion_synthesis import build_collusion_log


def test_colluding_pairs_are_caught_with_no_false_positives() -> None:
    records, truth = build_collusion_log(pairs=6, per_pair=8)
    report = evaluate_collusion(records, truth)
    assert report.catch_rate == 1.0
    assert report.false_positive_rate == 0.0
    assert report.auroc >= 0.9


def test_empty_log_yields_a_zeroed_report() -> None:
    report = evaluate_collusion([], {})
    assert report.pair_count == 0
    assert report.catch_rate == 0.0
