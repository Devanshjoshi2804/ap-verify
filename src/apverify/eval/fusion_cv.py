"""K-fold cross-validation for the fusion fit.

A single held-out split reports one number from one arbitrary partition — on a few
dozen rows that number swings with the split. K-fold rotation evaluates every row
exactly once on a model that never trained on it, then reports the mean across folds
**and the spread**, so the headline carries its own uncertainty instead of hiding it.

The interval is a Student-t 95% CI over the per-fold scores (small k, so t not z). It
answers the only honest question about a small-n metric: how much would this move on
another sample?
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from apverify.eval.calibration import expected_calibration_error
from apverify.eval.fusion import FeatureRow, auroc, fit_logistic

# Two-sided 95% t critical values by degrees of freedom (folds - 1). Folds are few, so
# the normal approximation under-covers; beyond the table the difference is negligible.
_T_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}
_Z_95 = 1.960


@dataclass(frozen=True, slots=True)
class CrossValidatedMetric:
    name: str
    per_fold: tuple[float, ...]
    mean: float
    std: float  # sample standard deviation across folds
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class FusionCrossValidation:
    folds: int  # folds actually evaluated (non-empty test sets)
    row_count: int
    auroc: CrossValidatedMetric
    ece: CrossValidatedMetric
    primaries: tuple[str, ...]  # >1 ⇒ rows pooled across extractors, unsafe to read as one


def summarize_metric(name: str, values: Sequence[float]) -> CrossValidatedMetric:
    """Mean, sample std, and a Student-t 95% CI over per-fold scores."""
    scores = tuple(values)
    if not scores:
        return CrossValidatedMetric(name, (), 0.0, 0.0, 0.0, 0.0)
    mean = sum(scores) / len(scores)
    if len(scores) < 2:
        return CrossValidatedMetric(name, scores, mean, 0.0, mean, mean)
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    std = math.sqrt(variance)
    critical = _T_95.get(len(scores) - 1, _Z_95)
    half_width = critical * std / math.sqrt(len(scores))
    return CrossValidatedMetric(name, scores, mean, std, mean - half_width, mean + half_width)


def fold_split(
    rows: Sequence[FeatureRow], folds: int, fold: int
) -> tuple[list[FeatureRow], list[FeatureRow]]:
    """Interleaved k-fold split: the held-out fold is every ``folds``-th row from
    ``fold``, the rest train. Interleaving (not contiguous blocks) keeps each fold
    spanning the whole dataset."""
    train = [row for index, row in enumerate(rows) if index % folds != fold]
    test = [row for index, row in enumerate(rows) if index % folds == fold]
    return train, test


def cross_validate_fusion(rows: Sequence[FeatureRow], folds: int = 5) -> FusionCrossValidation:
    """Fit fusion on k-1 folds, score the fused signal on the held-out fold, repeat -
    so every row is scored once by a model blind to it, with fold-to-fold spread."""
    aurocs: list[float] = []
    eces: list[float] = []
    for fold in range(folds):
        train, test = fold_split(rows, folds, fold)
        if not test:
            continue
        model = fit_logistic(train)
        samples = [(model.predict_proba(row.features()), row.correct) for row in test]
        aurocs.append(auroc(samples))
        eces.append(expected_calibration_error(samples))

    return FusionCrossValidation(
        folds=len(aurocs),
        row_count=len(rows),
        auroc=summarize_metric("fused AUROC", aurocs),
        ece=summarize_metric("fused ECE", eces),
        primaries=tuple(sorted({row.primary for row in rows if row.primary})),
    )
