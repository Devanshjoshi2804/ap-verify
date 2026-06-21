"""Document-level auto-approval gate — the money metric.

Field-level trust answers "can this field be trusted?"; accounts payable asks the
question one level up: *can this whole invoice be auto-posted without a human?* A
document is only auto-approvable if **every** field on it is right, so its trust is
bounded by its weakest field — one shaky line and the invoice needs review.

This aggregates the per-field fused trust into one score per document (the weakest
field) against one label per document (were *all* fields correct), then reuses the
exact risk-coverage machinery the field gate uses: the operating point is the
threshold that auto-approves the most invoices at zero wrong ones.

The fusion model is fit on a *document*-disjoint split — every field of a document
sits on one side of the train/test divide, so a model is never evaluated on an
invoice it partly trained on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.eval.calibration import (
    CoveragePoint,
    Sample,
    best_operating_point,
    expected_calibration_error,
    operating_point_at,
    risk_coverage,
)
from apverify.eval.fusion import FeatureRow, LogisticRegression, auroc, fit_logistic

# Error budgets reported on the autonomy curve: the strict provably-safe point, then
# the realistic operating points an AP team might actually run at.
DEFAULT_AUTONOMY_BUDGETS = (0.0, 0.01, 0.02, 0.05)


@dataclass(frozen=True, slots=True)
class DocumentGateEvaluation:
    train_documents: int
    test_documents: int
    field_count: int  # fields across the test documents — the n behind each decision
    auroc: float
    ece: float
    operating_point: CoveragePoint
    curve: tuple[CoveragePoint, ...]
    autonomy: tuple[tuple[float, CoveragePoint], ...]  # (error budget, best point within it)
    primaries: tuple[str, ...]  # >1 ⇒ rows pooled across extractors, unsafe to read as one


def aggregate_document(field_trust: Sequence[float], field_correct: Sequence[bool]) -> Sample:
    """Collapse a document's fields into one ``(trust, correct)`` decision: trust is the
    weakest field (the document is only as safe as its shakiest value), correct holds
    only if every field is right (one wrong field means the invoice was misread)."""
    return (min(field_trust), all(field_correct))


def split_documents(
    rows: Sequence[FeatureRow], test_fraction: float = 0.3
) -> tuple[list[FeatureRow], list[FeatureRow]]:
    """Partition rows by *document* so no invoice straddles the train/test divide.

    Documents are split, not rows: a model evaluated on an invoice whose other fields
    trained it would report a leaked, optimistic number. Deterministic interleaving over
    sorted labels keeps both sides spanning the dataset."""
    labels = sorted({row.label for row in rows})
    stride = max(2, round(1.0 / test_fraction))
    test_labels = {label for index, label in enumerate(labels) if index % stride == 0}
    train = [row for row in rows if row.label not in test_labels]
    test = [row for row in rows if row.label in test_labels]
    return train, test


def document_samples(rows: Sequence[FeatureRow], model: LogisticRegression) -> list[Sample]:
    """One ``(trust, correct)`` sample per document, in sorted-label order."""
    by_document: dict[str, list[FeatureRow]] = {}
    for row in rows:
        by_document.setdefault(row.label, []).append(row)
    return [
        aggregate_document(
            [model.predict_proba(row.features()) for row in fields],
            [row.correct for row in fields],
        )
        for _, fields in sorted(by_document.items())
    ]


def autonomy_curve(
    samples: Sequence[Sample], budgets: Sequence[float] = DEFAULT_AUTONOMY_BUDGETS
) -> tuple[tuple[float, CoveragePoint], ...]:
    """For each error budget, the most invoices auto-approvable within it — the
    selective-autonomy curve: how coverage grows as a little error is tolerated."""
    return tuple((budget, operating_point_at(samples, max_error=budget)) for budget in budgets)


def evaluate_document_gate(
    rows: Sequence[FeatureRow], test_fraction: float = 0.3
) -> DocumentGateEvaluation:
    """Fit fusion on document-disjoint training rows, then score the auto-approval gate
    on held-out invoices — the share auto-postable at the zero-error operating point."""
    train, test = split_documents(rows, test_fraction)
    model = fit_logistic(train)
    samples = document_samples(test, model)
    return DocumentGateEvaluation(
        train_documents=len({row.label for row in train}),
        test_documents=len({row.label for row in test}),
        field_count=len(test),
        auroc=auroc(samples),
        ece=expected_calibration_error(samples),
        operating_point=best_operating_point(samples),
        curve=tuple(risk_coverage(samples)),
        autonomy=autonomy_curve(samples),
        primaries=tuple(sorted({row.primary for row in rows if row.primary})),
    )
