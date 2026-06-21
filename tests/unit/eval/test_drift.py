from __future__ import annotations

from pathlib import Path

from apverify.eval.drift import compare, load_snapshot, save_snapshot
from apverify.eval.metrics import EvalSnapshot
from apverify.eval.runner import run_eval


def _snapshot(
    catch: float, false_hold: float = 0.0, kinds: dict[str, float] | None = None
) -> EvalSnapshot:
    return EvalSnapshot(
        clean_count=10,
        corrupt_count=60,
        catch_rate=catch,
        false_hold_rate=false_hold,
        safe_auto_approval_rate=1.0,
        escaped=0,
        per_kind=kinds or {"flip_total_digit": catch},
    )


def test_snapshot_round_trips_through_json(tmp_path: Path) -> None:
    snapshot = run_eval(5).to_snapshot()
    path = tmp_path / "baseline.json"
    save_snapshot(snapshot, path)

    assert load_snapshot(path) == snapshot


def test_identical_runs_show_no_regression() -> None:
    base = _snapshot(1.0)
    assert compare(base, _snapshot(1.0)).regressed is False


def test_dropped_catch_rate_is_a_regression() -> None:
    drift = compare(_snapshot(1.0), _snapshot(0.8))

    assert drift.regressed is True
    assert drift.catch_rate_delta < 0
    assert any("catch rate fell" in reason for reason in drift.reasons)


def test_risen_false_hold_is_a_regression() -> None:
    drift = compare(_snapshot(1.0, false_hold=0.0), _snapshot(1.0, false_hold=0.1))

    assert drift.regressed is True
    assert any("false-hold" in reason for reason in drift.reasons)


def test_improvement_is_not_a_regression() -> None:
    drift = compare(_snapshot(0.9), _snapshot(1.0))

    assert drift.regressed is False
    assert drift.catch_rate_delta > 0


def test_tolerance_absorbs_small_dips() -> None:
    drift = compare(_snapshot(1.0), _snapshot(0.98), catch_tolerance=0.05)
    assert drift.regressed is False
