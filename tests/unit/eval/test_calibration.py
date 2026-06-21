from __future__ import annotations

import pytest

from apverify.eval.calibration import (
    apply_temperature,
    best_operating_point,
    expected_calibration_error,
    fit_temperature,
    operating_point_at,
    reliability_bins,
    risk_coverage,
    temperature_scaled,
)


def test_perfectly_calibrated_has_zero_ece() -> None:
    samples = [(1.0, True), (1.0, True), (0.0, False), (0.0, False)]
    assert expected_calibration_error(samples) == pytest.approx(0.0)


def test_overconfident_wrong_predictions_raise_ece() -> None:
    samples = [(0.9, False)] * 10  # claims 0.9, never right
    assert expected_calibration_error(samples) == pytest.approx(0.9)


def test_reliability_bins_summarise_each_bucket() -> None:
    bins = reliability_bins([(0.95, True), (0.95, False)], bins=10)
    top = next(b for b in bins if b.lower == 0.9)
    assert top.count == 2
    assert top.accuracy == pytest.approx(0.5)


def test_risk_coverage_trades_coverage_for_error() -> None:
    samples = [(0.9, True), (0.5, False)]
    points = {round(p.threshold, 2): p for p in risk_coverage(samples, steps=10)}
    assert points[0.0].coverage == pytest.approx(1.0)
    assert points[0.0].error == pytest.approx(0.5)
    assert points[0.7].coverage == pytest.approx(0.5)
    assert points[0.7].error == pytest.approx(0.0)


def test_operating_point_maximises_coverage_at_zero_error() -> None:
    samples = [(0.9, True), (0.8, True), (0.4, False)]
    point = best_operating_point(samples)
    assert point.error == pytest.approx(0.0)
    assert point.coverage == pytest.approx(2 / 3)


def test_no_error_free_threshold_when_a_wrong_prediction_is_max_confidence() -> None:
    # The honest case: a wrong extraction at confidence 1.0 means no threshold is
    # error-free, so the operating point must report a non-zero error.
    point = best_operating_point([(1.0, True), (1.0, False)])
    assert point.error > 0.0


def test_error_budget_buys_more_coverage_than_zero_error() -> None:
    # A small wrong prediction sits just below the clean ones; allowing a modest error
    # budget should cover it and raise coverage above the zero-error operating point.
    samples = [(0.9, True), (0.8, True), (0.7, False), (0.6, True)]
    strict = operating_point_at(samples, max_error=0.0)
    lenient = operating_point_at(samples, max_error=0.30)
    assert lenient.coverage > strict.coverage
    assert lenient.error <= 0.30


def test_operating_point_at_zero_matches_best_operating_point() -> None:
    samples = [(0.9, True), (0.8, True), (0.4, False)]
    assert operating_point_at(samples, max_error=0.0) == best_operating_point(samples)


def test_temperature_one_is_the_identity() -> None:
    assert apply_temperature(0.8, 1.0) == pytest.approx(0.8)


def test_high_temperature_softens_overconfidence() -> None:
    softened = apply_temperature(0.99, 3.0)
    assert 0.5 < softened < 0.99


def test_fitting_temperature_reduces_calibration_error() -> None:
    # Overconfident: claims 0.99 but only 70% right.
    samples = [(0.99, True)] * 70 + [(0.99, False)] * 30
    before = expected_calibration_error(samples)

    temperature = fit_temperature(samples)
    after = expected_calibration_error(temperature_scaled(samples, temperature))

    assert temperature > 1.0  # softening
    assert after < before
    assert after < 0.05  # recalibrated close to the true 0.70
