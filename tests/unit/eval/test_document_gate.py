from __future__ import annotations

from collections import Counter

from apverify.eval.document_gate import (
    aggregate_document,
    autonomy_curve,
    document_samples,
    evaluate_document_gate,
    split_documents,
)
from apverify.eval.fusion import FeatureRow, LogisticRegression


def _row(
    label: str,
    field: str,
    *,
    arithmetic: bool,
    correct: bool,
    primary: str = "gemini",
) -> FeatureRow:
    # Every signal is held constant except ``arithmetic_passed``, so correctness is a
    # clean function of one feature and a fitted model separates the rows perfectly.
    return FeatureRow(
        label=label,
        field=field,
        critic_confidence=0.8,
        verbalized_confidence=0.5,
        cross_check_passed=True,
        arithmetic_passed=arithmetic,
        format_passed=True,
        cross_model_agrees=True,
        correct=correct,
        primary=primary,
    )


def test_document_trust_is_its_weakest_field() -> None:
    # The least-trustworthy field bounds the whole document.
    sample = aggregate_document([0.91, 0.40, 0.83], [True, True, True])
    assert sample == (0.40, True)


def test_a_document_is_correct_only_if_every_field_is() -> None:
    # One wrong field sinks the document, even if it is not the weakest-scored.
    sample = aggregate_document([0.91, 0.83], [True, False])
    assert sample == (0.83, False)


def test_split_keeps_each_document_whole_and_on_one_side() -> None:
    rows = [
        _row(label, "total", arithmetic=True, correct=True) for label in ("a", "b", "c", "d")
    ] + [_row(label, "vendor", arithmetic=True, correct=True) for label in ("a", "b", "c", "d")]

    train, test = split_documents(rows, test_fraction=0.3)

    train_docs = {row.label for row in train}
    test_docs = {row.label for row in test}
    assert train_docs.isdisjoint(test_docs)  # no field of a document leaks across the split
    assert train_docs | test_docs == {"a", "b", "c", "d"}
    # Both fields of every document stay together on whichever side it landed.
    for side in (train, test):
        per_document = Counter(row.label for row in side)
        assert all(count == 2 for count in per_document.values())


def test_autonomy_curve_reports_more_coverage_as_the_error_budget_loosens() -> None:
    # One wrong document sits among clean ones at a slightly lower trust; a looser
    # budget should auto-approve strictly more invoices.
    samples = [(0.9, True), (0.8, True), (0.7, False), (0.6, True)]
    curve = dict(autonomy_curve(samples, budgets=(0.0, 0.3)))

    assert curve[0.3].coverage >= curve[0.0].coverage
    assert all(point.error <= budget for budget, point in curve.items())


def test_document_samples_groups_fields_per_document() -> None:
    rows = [
        _row("doc1", "total", arithmetic=True, correct=True),
        _row("doc1", "vendor", arithmetic=True, correct=False),
        _row("doc2", "total", arithmetic=True, correct=True),
    ]
    flat = LogisticRegression(weights=(0.0,) * 6, bias=0.0)  # every field scores 0.5

    samples = document_samples(rows, flat)

    assert samples == [(0.5, False), (0.5, True)]  # doc1 has a wrong field, doc2 is clean


def test_gate_auto_approves_clean_documents_with_no_error() -> None:
    good = [
        row
        for index in range(4)
        for row in (
            _row(f"{index}-good", "total", arithmetic=True, correct=True),
            _row(f"{index}-good", "vendor", arithmetic=True, correct=True),
        )
    ]
    bad = [
        row
        for index in range(4)
        for row in (
            _row(f"{index}-zbad", "total", arithmetic=True, correct=True),
            _row(f"{index}-zbad", "vendor", arithmetic=False, correct=False),
        )
    ]

    evaluation = evaluate_document_gate(good + bad, test_fraction=0.3)

    assert evaluation.train_documents + evaluation.test_documents == 8
    assert evaluation.test_documents > 0
    # Arithmetic perfectly predicts correctness, so the doc gate separates cleanly and
    # there is an error-free threshold that still auto-approves the clean documents.
    assert evaluation.auroc == 1.0
    assert evaluation.operating_point.error == 0.0
    assert evaluation.operating_point.coverage > 0.0
    assert evaluation.primaries == ("gemini",)


def test_empty_rows_yield_an_empty_evaluation() -> None:
    evaluation = evaluate_document_gate([])

    assert evaluation.train_documents == 0
    assert evaluation.test_documents == 0
    assert evaluation.operating_point.coverage == 0.0
