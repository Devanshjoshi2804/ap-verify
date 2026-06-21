"""Drift detection — has reliability silently degraded since a baseline?

A model swap or a prompt change can quietly lower the catch rate or start holding
clean invoices. Comparing a fresh run against a saved baseline snapshot turns that
silent regression into a loud, gating signal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from apverify.eval.metrics import EvalSnapshot


@dataclass(frozen=True, slots=True)
class DriftReport:
    catch_rate_delta: float
    false_hold_delta: float
    safe_auto_delta: float
    per_kind_delta: dict[str, float]
    regressed: bool
    reasons: tuple[str, ...]


def compare(
    baseline: EvalSnapshot,
    candidate: EvalSnapshot,
    catch_tolerance: float = 0.0,
    false_hold_tolerance: float = 0.0,
) -> DriftReport:
    catch_delta = candidate.catch_rate - baseline.catch_rate
    false_hold_delta = candidate.false_hold_rate - baseline.false_hold_rate
    safe_auto_delta = candidate.safe_auto_approval_rate - baseline.safe_auto_approval_rate

    per_kind_delta = {
        kind: candidate.per_kind.get(kind, 0.0) - rate for kind, rate in baseline.per_kind.items()
    }

    reasons: list[str] = []
    if catch_delta < -catch_tolerance:
        reasons.append(
            f"catch rate fell {-catch_delta:.1%} "
            f"({baseline.catch_rate:.1%} → {candidate.catch_rate:.1%})"
        )
    if false_hold_delta > false_hold_tolerance:
        reasons.append(f"false-hold rate rose {false_hold_delta:.1%}")
    reasons.extend(
        f"{kind} catch fell {-delta:.0%}"
        for kind, delta in per_kind_delta.items()
        if delta < -catch_tolerance
    )

    return DriftReport(
        catch_rate_delta=catch_delta,
        false_hold_delta=false_hold_delta,
        safe_auto_delta=safe_auto_delta,
        per_kind_delta=per_kind_delta,
        regressed=bool(reasons),
        reasons=tuple(reasons),
    )


def save_snapshot(snapshot: EvalSnapshot, path: Path) -> None:
    path.write_text(json.dumps(asdict(snapshot), indent=2))


def load_snapshot(path: Path) -> EvalSnapshot:
    return EvalSnapshot(**json.loads(path.read_text()))
