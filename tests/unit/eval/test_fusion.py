from __future__ import annotations

import pytest

from apverify.eval.fusion import (
    FEATURES,
    FeatureRow,
    LogisticRegression,
    auroc,
    disagreement_catch_rate,
    evaluate_fusion,
    explain_fusion,
    fit_logistic,
    merge_rows,
    row_from_dict,
    row_to_dict,
    train_test_split,
)


def _xrow(**over: object) -> FeatureRow:
    base: dict[str, object] = {
        "label": "d",
        "field": "total",
        "critic_confidence": 0.5,
        "verbalized_confidence": 0.5,
        "cross_check_passed": True,
        "arithmetic_passed": False,
        "format_passed": True,
        "cross_model_agrees": True,
        "correct": False,
    }
    base.update(over)
    return FeatureRow(**base)  # type: ignore[arg-type]


def test_explain_fusion_contribution_is_weight_times_feature() -> None:
    model = LogisticRegression(weights=(0.0, 0.0, 0.0, 2.0, 0.0, 0.0), bias=0.0)
    failed = explain_fusion(model, _xrow(arithmetic_passed=False))
    assert next(f for f in failed.factors if f.signal == "arithmetic_passed").contribution == 0.0
    passed = explain_fusion(model, _xrow(arithmetic_passed=True))
    assert next(f for f in passed.factors if f.signal == "arithmetic_passed").contribution == 2.0


def test_explain_fusion_ranks_the_dominant_signal_first() -> None:
    model = LogisticRegression(weights=(0.1, 0.0, 0.0, 3.0, 0.0, 0.0), bias=0.0)
    result = explain_fusion(model, _xrow(arithmetic_passed=True, critic_confidence=1.0))
    assert result.factors[0].signal == "arithmetic_passed"


def _row(
    *,
    label: str = "doc",
    field: str = "total",
    critic: float = 1.0,
    verbalized: float = 0.95,
    cross_check: bool = True,
    arithmetic: bool = True,
    fmt: bool = True,
    agrees: bool = True,
    correct: bool = True,
    primary: str = "gemini",
) -> FeatureRow:
    return FeatureRow(
        label=label,
        field=field,
        critic_confidence=critic,
        verbalized_confidence=verbalized,
        cross_check_passed=cross_check,
        arithmetic_passed=arithmetic,
        format_passed=fmt,
        cross_model_agrees=agrees,
        correct=correct,
        primary=primary,
        secondary="ollama",
    )


def test_auroc_rewards_separation() -> None:
    perfect = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    assert auroc(perfect) == pytest.approx(1.0)
    reversed_scores = [(0.1, True), (0.2, True), (0.8, False), (0.9, False)]
    assert auroc(reversed_scores) == pytest.approx(0.0)


def test_auroc_is_half_without_both_classes_or_under_ties() -> None:
    assert auroc([(0.9, True), (0.8, True)]) == pytest.approx(0.5)
    assert auroc([(0.5, True), (0.5, False)]) == pytest.approx(0.5)


def test_fusion_learns_an_informative_signal() -> None:
    # cross-model agreement perfectly separates correct from wrong here; a useless
    # high verbalized confidence is constant. The model should lean on agreement.
    rows = [_row(agrees=True, correct=True) for _ in range(20)]
    rows += [_row(agrees=False, correct=False) for _ in range(20)]

    model = fit_logistic(rows)
    scores = [(model.predict_proba(r.features()), r.correct) for r in rows]

    assert auroc(scores) == pytest.approx(1.0)
    assert model.coefficients["cross_model_agrees"] > model.coefficients["verbalized_confidence"]
    assert set(model.coefficients) == set(FEATURES)


def test_predict_proba_stays_in_unit_interval() -> None:
    model = fit_logistic([_row(correct=True), _row(agrees=False, correct=False)])
    assert 0.0 <= model.predict_proba(_row().features()) <= 1.0


def test_train_test_split_is_deterministic_and_disjoint() -> None:
    rows = [_row(label=str(i)) for i in range(10)]
    train, test = train_test_split(rows, test_fraction=0.3)

    assert {r.label for r in train}.isdisjoint({r.label for r in test})
    assert len(train) + len(test) == len(rows)
    assert train_test_split(rows, test_fraction=0.3)[1] == test  # stable


def test_disagreement_catch_rate_counts_confidently_wrong_caught() -> None:
    rows = [
        _row(critic=1.0, agrees=False, correct=False),  # confidently wrong, caught
        _row(critic=1.0, agrees=True, correct=False),  # confidently wrong, missed
        _row(critic=0.3, agrees=False, correct=False),  # low confidence, ignored
        _row(critic=1.0, agrees=True, correct=True),  # correct, ignored
    ]
    catch = disagreement_catch_rate(rows, confidence_threshold=0.9)

    assert catch.high_confidence_errors == 2
    assert catch.caught_by_disagreement == 1
    assert catch.catch_rate == pytest.approx(0.5)


def test_disagreement_catch_rate_is_zero_without_high_confidence_errors() -> None:
    assert disagreement_catch_rate([_row(correct=True)]).catch_rate == 0.0


def test_evaluation_calibrates_the_fused_score_without_changing_its_ranking() -> None:
    rows = []
    for i in range(60):
        correct = i % 2 == 0  # alternating, coprime with the stride-3 split → both classes
        rows.append(_row(label=str(i), agrees=correct, arithmetic=correct, correct=correct))

    evaluation = evaluate_fusion(rows)
    names = [s.name for s in evaluation.signals]
    fused = next(s for s in evaluation.signals if s.name == "fused")
    calibrated = next(s for s in evaluation.signals if s.name == "fused + T-scale")

    assert "fused + T-scale" in names
    assert evaluation.temperature > 0.0
    # temperature scaling is monotonic — discrimination is identical, calibration no worse
    assert calibrated.auroc == pytest.approx(fused.auroc)
    assert calibrated.ece <= fused.ece + 1e-9


def test_evaluation_exposes_a_single_primary_and_the_cross_model_signal() -> None:
    rows = [_row(label=str(i), correct=i % 2 == 0, primary="gemini") for i in range(20)]
    evaluation = evaluate_fusion(rows)

    assert evaluation.primaries == ("gemini",)
    assert "cross-model" in [s.name for s in evaluation.signals]


def test_evaluation_flags_pooled_primaries_as_unsafe() -> None:
    rows = [_row(label=str(i), correct=i % 2 == 0, primary="gemini") for i in range(10)]
    rows += [_row(label=f"g{i}", correct=i % 2 == 0, primary="groq") for i in range(10)]

    assert evaluate_fusion(rows).primaries == ("gemini", "groq")  # surfaced for the warning


def test_feature_row_round_trips_extractor_identity() -> None:
    row = _row(primary="gemini")
    assert row_from_dict(row_to_dict(row)).primary == "gemini"
    assert row_from_dict(row_to_dict(row)).secondary == "ollama"


def test_merge_accumulates_and_refreshes_by_document_and_field() -> None:
    existing = [_row(label="a", field="total", correct=True)]
    fresh = [
        _row(label="a", field="total", correct=False),  # same key → refreshed
        _row(label="b", field="total", correct=True),  # new key → added
    ]
    merged = merge_rows(existing, fresh)

    assert len(merged) == 2
    refreshed = next(r for r in merged if r.label == "a")
    assert refreshed.correct is False
