from __future__ import annotations

import math

import pytest

from apverify.eval.uncertainty import (
    cluster_values,
    confidence_from_entropy,
    self_consistency,
    semantic_entropy,
)


def _amount_equivalent(left: str, right: str) -> bool:
    return float(left) == float(right)


def test_full_agreement_is_consistent_and_zero_entropy() -> None:
    values = ["118", "118", "118", "118"]
    assert self_consistency(values) == pytest.approx(1.0)
    assert semantic_entropy(values) == pytest.approx(0.0)


def test_total_disagreement_maximises_entropy() -> None:
    values = ["a", "b", "c", "d"]
    assert self_consistency(values) == pytest.approx(0.25)
    assert semantic_entropy(values) == pytest.approx(math.log(4))


def test_semantic_clustering_ignores_formatting_not_meaning() -> None:
    values = ["184200", "184200.00", "184200.0", "999"]
    # raw strings would look 4-way split; by meaning it is 3 agree + 1 outlier
    assert self_consistency(values, _amount_equivalent) == pytest.approx(0.75)
    assert semantic_entropy(values, _amount_equivalent) < semantic_entropy(values)


def test_cluster_values_groups_by_equivalence() -> None:
    clusters = cluster_values(["100", "100.0", "200"], _amount_equivalent)
    assert sorted(len(c) for c in clusters) == [1, 2]


def test_confidence_from_entropy_is_monotonic_and_bounded() -> None:
    assert confidence_from_entropy(0.0) == pytest.approx(1.0)
    assert 0.0 < confidence_from_entropy(math.log(4)) < 1.0


def test_empty_samples_are_neutral() -> None:
    assert self_consistency([]) == 0.0
    assert semantic_entropy([]) == 0.0
