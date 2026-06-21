from __future__ import annotations

from apverify.eval.collusion_synthesis import build_collusion_log


def test_log_has_both_colluding_and_normal_pairs() -> None:
    _, truth = build_collusion_log(pairs=6, per_pair=8)
    assert any(truth.values())  # at least one colluding pair
    assert not all(truth.values())  # and at least one normal pair


def test_every_truth_pair_has_enough_records_to_score() -> None:
    records, truth = build_collusion_log(pairs=4, per_pair=8)
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.approver, record.vendor)
        counts[key] = counts.get(key, 0) + 1
    for pair in truth:
        assert counts[pair] >= 3
