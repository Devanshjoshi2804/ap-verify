"""Multi-signal fusion — combine independent trust signals into one calibrated
``P(correct)`` per field.

A single confidence number hides two distinct properties, and our own measurements
show they come apart: a signal can be well *calibrated* (0.9 means 90%) yet unable
to *discriminate* (it says 0.95 for everything, so no threshold separates right from
wrong). Fusion's job is a score that has both.

The inputs are chosen to fail *differently*:

* ``critic_confidence`` — the critic's structural score from deterministic checks.
* ``verbalized_confidence`` — the model's own self-report; blind in exactly the
  places the model itself is wrong, since it comes from that same model.
* ``cross_check`` / ``arithmetic`` / ``format`` — the critic's individual checks.
* ``cross_model_agrees`` — a *second* model reading the same page. Independent of
  both the extractor and the critic, it is the only signal that can catch a
  confidently-wrong-but-self-consistent extraction the others share a blind spot
  for.

The fusion model is a plain logistic regression: interpretable (each signal's
weight is inspectable), dependency-free, and deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from apverify.domain.explanation import Explanation, Factor, explanation
from apverify.eval.calibration import (
    CoveragePoint,
    Sample,
    best_operating_point,
    expected_calibration_error,
    fit_temperature,
    temperature_scaled,
)

FEATURES: tuple[str, ...] = (
    "critic_confidence",
    "verbalized_confidence",
    "cross_check_passed",
    "arithmetic_passed",
    "format_passed",
    "cross_model_agrees",
)


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """One field of one document: its signals plus whether it was actually correct.

    ``primary`` records which extractor produced the row. A fusion model is fit against
    one extractor's error distribution and confidence calibration, so rows from
    different primaries must never be pooled into one fit — the identity is carried here
    so the evaluation can enforce that.
    """

    label: str
    field: str
    critic_confidence: float
    verbalized_confidence: float
    cross_check_passed: bool
    arithmetic_passed: bool
    format_passed: bool
    cross_model_agrees: bool
    correct: bool
    primary: str = ""
    secondary: str = ""

    def features(self) -> list[float]:
        return [
            self.critic_confidence,
            self.verbalized_confidence,
            float(self.cross_check_passed),
            float(self.arithmetic_passed),
            float(self.format_passed),
            float(self.cross_model_agrees),
        ]


def row_to_dict(row: FeatureRow) -> dict[str, object]:
    return {
        "label": row.label,
        "field": row.field,
        "critic_confidence": row.critic_confidence,
        "verbalized_confidence": row.verbalized_confidence,
        "cross_check_passed": row.cross_check_passed,
        "arithmetic_passed": row.arithmetic_passed,
        "format_passed": row.format_passed,
        "cross_model_agrees": row.cross_model_agrees,
        "correct": row.correct,
        "primary": row.primary,
        "secondary": row.secondary,
    }


def row_from_dict(payload: dict[str, object]) -> FeatureRow:
    return FeatureRow(
        label=str(payload["label"]),
        field=str(payload["field"]),
        critic_confidence=float(payload["critic_confidence"]),  # type: ignore[arg-type]
        verbalized_confidence=float(payload["verbalized_confidence"]),  # type: ignore[arg-type]
        cross_check_passed=bool(payload["cross_check_passed"]),
        arithmetic_passed=bool(payload["arithmetic_passed"]),
        format_passed=bool(payload["format_passed"]),
        cross_model_agrees=bool(payload["cross_model_agrees"]),
        correct=bool(payload["correct"]),
        primary=str(payload.get("primary", "")),
        secondary=str(payload.get("secondary", "")),
    )


def merge_rows(existing: Sequence[FeatureRow], fresh: Sequence[FeatureRow]) -> list[FeatureRow]:
    """Accumulate rows across runs, keyed by ``(document, field)`` so re-extracting a
    document refreshes its rows rather than double-counting them. Free-tier quota caps
    a single run, so a usable n is reached by appending several."""
    merged = {(row.label, row.field): row for row in existing}
    for row in fresh:
        merged[(row.label, row.field)] = row
    return list(merged.values())


@dataclass(frozen=True, slots=True)
class LogisticRegression:
    weights: tuple[float, ...]
    bias: float

    def predict_proba(self, features: Sequence[float]) -> float:
        z = self.bias + sum(w * x for w, x in zip(self.weights, features, strict=True))
        return _sigmoid(z)

    @property
    def coefficients(self) -> dict[str, float]:
        """Each signal's learned weight — the model's explanation of itself."""
        return dict(zip(FEATURES, self.weights, strict=True))


def fit_logistic(
    rows: Sequence[FeatureRow],
    *,
    learning_rate: float = 0.3,
    epochs: int = 4000,
    l2: float = 1.0,
) -> LogisticRegression:
    """Batch gradient descent with L2 regularisation — deterministic, no optimiser
    dependency. L2 keeps a perfectly-separating signal from blowing its weight up to
    infinity, which would overfit the handful of rows that separate cleanly."""
    if not rows:
        return LogisticRegression(weights=tuple(0.0 for _ in FEATURES), bias=0.0)

    dimension = len(FEATURES)
    weights = [0.0] * dimension
    bias = 0.0
    n = len(rows)
    matrix = [row.features() for row in rows]
    targets = [1.0 if row.correct else 0.0 for row in rows]

    for _ in range(epochs):
        weight_gradient = [0.0] * dimension
        bias_gradient = 0.0
        for features, target in zip(matrix, targets, strict=True):
            error = _sigmoid(bias + sum(w * x for w, x in zip(weights, features, strict=True)))
            error -= target
            for j in range(dimension):
                weight_gradient[j] += error * features[j]
            bias_gradient += error
        for j in range(dimension):
            weights[j] -= learning_rate * (weight_gradient[j] / n + l2 * weights[j] / n)
        bias -= learning_rate * (bias_gradient / n)

    return LogisticRegression(weights=tuple(weights), bias=bias)


def train_test_split(
    rows: Sequence[FeatureRow], test_fraction: float = 0.3
) -> tuple[list[FeatureRow], list[FeatureRow]]:
    """A deterministic interleaved split, so a learned model is never evaluated on
    the rows it trained on. Interleaving (rather than a head/tail cut) keeps both
    splits spanning the whole dataset."""
    stride = max(2, round(1.0 / test_fraction))
    train = [row for index, row in enumerate(rows) if index % stride != 0]
    test = [row for index, row in enumerate(rows) if index % stride == 0]
    return train, test


@dataclass(frozen=True, slots=True)
class DisagreementCatch:
    high_confidence_errors: int
    caught_by_disagreement: int

    @property
    def catch_rate(self) -> float:
        if not self.high_confidence_errors:
            return 0.0
        return self.caught_by_disagreement / self.high_confidence_errors


def disagreement_catch_rate(
    rows: Sequence[FeatureRow], confidence_threshold: float = 0.9
) -> DisagreementCatch:
    """The diagnostic that gates fusion: of the extractions the critic is confident
    in but got *wrong*, how many does cross-model disagreement already flag? A high
    rate means an independent signal can reach the error-free threshold the critic
    alone cannot."""
    high_confidence_errors = [
        row for row in rows if row.critic_confidence >= confidence_threshold and not row.correct
    ]
    caught = [row for row in high_confidence_errors if not row.cross_model_agrees]
    return DisagreementCatch(
        high_confidence_errors=len(high_confidence_errors),
        caught_by_disagreement=len(caught),
    )


def auroc(samples: Sequence[tuple[float, bool]]) -> float:
    """Area under the ROC curve — the discrimination axis ECE misses.

    The probability that a randomly chosen correct field scores above a randomly
    chosen wrong one. 0.5 is no better than chance; 1.0 is perfect separation.
    Computed via the rank-sum identity, with ties counted as half.
    """
    positives = [score for score, correct in samples if correct]
    negatives = [score for score, correct in samples if not correct]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


@dataclass(frozen=True, slots=True)
class SignalMetrics:
    name: str
    sample_count: int
    auroc: float  # discrimination: can it separate right from wrong?
    ece: float  # calibration: are the probabilities honest?
    operating_point: CoveragePoint


@dataclass(frozen=True, slots=True)
class FusionEvaluation:
    train_size: int
    test_size: int
    coefficients: dict[str, float]
    diagnostic: DisagreementCatch
    signals: tuple[SignalMetrics, ...]
    temperature: float
    primaries: tuple[str, ...]  # which extractor(s) produced the rows; >1 ⇒ unsafe to pool


def _signal_metrics(name: str, samples: Sequence[Sample]) -> SignalMetrics:
    return SignalMetrics(
        name=name,
        sample_count=len(samples),
        auroc=auroc(samples),
        ece=expected_calibration_error(samples),
        operating_point=best_operating_point(samples),
    )


def evaluate_fusion(rows: Sequence[FeatureRow], test_fraction: float = 0.3) -> FusionEvaluation:
    """Train the fused model on one split and score every signal on the *other*.

    Each signal — including the learned fusion — is measured on a held-out test set
    it never saw, so the reported numbers are not the overfit-on-its-own-training
    figure that the first question about any learned component asks after.
    """
    train, test = train_test_split(rows, test_fraction)
    model = fit_logistic(train)

    critic = [(row.critic_confidence, row.correct) for row in test]
    verbalized = [(row.verbalized_confidence, row.correct) for row in test]
    # The decorrelation diagnostic: does "the two models agree" by itself separate the
    # primary's correct extractions from its wrong ones? If this AUROC is ~0.5 the second
    # model's errors are correlated with the primary's and the signal is worthless.
    cross_model = [(float(row.cross_model_agrees), row.correct) for row in test]
    fused = [(model.predict_proba(row.features()), row.correct) for row in test]

    # Discriminate first, then calibrate: temperature is fit on the *training* fused
    # scores and applied to the test ones, so the calibrated number is honest. Scaling
    # is monotonic, so it lowers ECE without changing AUROC.
    train_fused = [(model.predict_proba(row.features()), row.correct) for row in train]
    temperature = fit_temperature(train_fused)
    fused_calibrated = temperature_scaled(fused, temperature)

    return FusionEvaluation(
        train_size=len(train),
        test_size=len(test),
        coefficients=model.coefficients,
        diagnostic=disagreement_catch_rate(rows),
        signals=(
            _signal_metrics("critic", critic),
            _signal_metrics("verbalized", verbalized),
            _signal_metrics("cross-model", cross_model),
            _signal_metrics("fused", fused),
            _signal_metrics("fused + T-scale", fused_calibrated),
        ),
        temperature=temperature,
        primaries=tuple(sorted({row.primary for row in rows if row.primary})),
    )


def explain_fusion(model: LogisticRegression, row: FeatureRow) -> Explanation:
    """Exact linear attribution: each signal's contribution to the log-odds of *correct*
    is ``weight * feature`` — SHAP for a linear model, in closed form. Positive raises
    P(correct); negative lowers it."""
    values = row.features()
    factors = [
        Factor(
            signal=name,
            value=f"{value:.2f}",
            contribution=weight * value,
            detail=f"weight {weight:+.2f} x {value:.2f}",
        )
        for name, weight, value in zip(FEATURES, model.weights, values, strict=True)
    ]
    factors.append(Factor("bias", "", model.bias, "model baseline"))
    probability = model.predict_proba(values)
    return explanation("fusion", f"P(correct) {probability:.2f}", factors)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)
