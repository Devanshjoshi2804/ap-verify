from __future__ import annotations

import pytest

from apverify.eval.fusion import FeatureRow
from apverify.eval.fusion_cv import (
    cross_validate_fusion,
    fold_split,
    summarize_metric,
)


def _row(label: str, *, arithmetic: bool, correct: bool, primary: str = "gemini") -> FeatureRow:
    # Correctness is a clean function of one feature, so a fitted model separates the
    # rows perfectly and every fold scores AUROC 1.0 — a deterministic CV fixture.
    return FeatureRow(
        label=label,
        field="total",
        critic_confidence=0.8,
        verbalized_confidence=0.5,
        cross_check_passed=True,
        arithmetic_passed=arithmetic,
        format_passed=True,
        cross_model_agrees=True,
        correct=correct,
        primary=primary,
    )


def test_summarize_metric_constant_has_zero_spread() -> None:
    metric = summarize_metric("fused AUROC", [0.8, 0.8, 0.8])
    assert metric.mean == pytest.approx(0.8)
    assert metric.std == pytest.approx(0.0)
    assert (metric.ci_low, metric.ci_high) == pytest.approx((0.8, 0.8))
    assert metric.per_fold == (0.8, 0.8, 0.8)


def test_summarize_metric_reports_dispersion_with_a_t_interval() -> None:
    # Two folds at 0.0 and 1.0: mean 0.5, sample std sqrt(0.5), 95% CI uses t(df=1)=12.706.
    metric = summarize_metric("fused AUROC", [0.0, 1.0])
    assert metric.mean == pytest.approx(0.5)
    assert metric.std == pytest.approx(0.7071, abs=1e-4)
    half_width = 12.706 * 0.7071 / 2**0.5
    assert metric.ci_low == pytest.approx(0.5 - half_width, abs=1e-3)
    assert metric.ci_high == pytest.approx(0.5 + half_width, abs=1e-3)


def test_summarize_metric_empty_is_zeroed() -> None:
    metric = summarize_metric("fused AUROC", [])
    assert metric.mean == 0.0
    assert metric.std == 0.0
    assert metric.per_fold == ()


def test_folds_partition_every_row_into_exactly_one_test_set() -> None:
    rows = [_row(str(i), arithmetic=True, correct=True) for i in range(10)]

    seen: set[str] = set()
    for fold in range(5):
        train, test = fold_split(rows, folds=5, fold=fold)
        train_labels = {row.label for row in train}
        test_labels = {row.label for row in test}
        assert train_labels.isdisjoint(test_labels)
        assert train_labels | test_labels == {str(i) for i in range(10)}
        assert seen.isdisjoint(test_labels)  # no row is tested twice
        seen |= test_labels
    assert seen == {str(i) for i in range(10)}  # every row tested once


def test_cross_validation_on_separable_data_is_perfect_and_stable() -> None:
    rows = [_row(str(i), arithmetic=(i % 2 == 0), correct=(i % 2 == 0)) for i in range(20)]

    result = cross_validate_fusion(rows, folds=5)

    assert result.folds == 5
    assert result.row_count == 20
    assert result.auroc.mean == pytest.approx(1.0)
    assert result.auroc.std == pytest.approx(0.0)
    assert result.auroc.ci_low == pytest.approx(1.0)
    assert result.primaries == ("gemini",)


def test_empty_rows_yield_a_zeroed_cross_validation() -> None:
    result = cross_validate_fusion([], folds=5)
    assert result.folds == 0
    assert result.row_count == 0
    assert result.auroc.mean == 0.0
