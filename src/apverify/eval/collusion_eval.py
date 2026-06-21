"""Score the collusion detector against a labelled approval log."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from apverify.domain.collusion import ApprovalRecord, CollusionSeverity, detect_collusion
from apverify.eval.fusion import auroc


@dataclass(frozen=True, slots=True)
class CollusionReport:
    pair_count: int
    colluding_count: int
    catch_rate: float
    false_positive_rate: float
    precision: float
    auroc: float


def evaluate_collusion(
    records: Sequence[ApprovalRecord], truth: dict[tuple[str, str], bool]
) -> CollusionReport:
    signals = detect_collusion(records)
    score_by_pair = {(signal.approver, signal.vendor): signal.score for signal in signals}
    flagged = {
        (signal.approver, signal.vendor)
        for signal in signals
        if signal.severity is not CollusionSeverity.NONE
    }

    colluding = [pair for pair, is_collusion in truth.items() if is_collusion]
    normal = [pair for pair, is_collusion in truth.items() if not is_collusion]
    flagged_truth = [pair for pair in truth if pair in flagged]
    samples = [(score_by_pair.get(pair, 0.0), is_collusion) for pair, is_collusion in truth.items()]

    return CollusionReport(
        pair_count=len(truth),
        colluding_count=len(colluding),
        catch_rate=_rate([p for p in colluding if p in flagged], colluding),
        false_positive_rate=_rate([p for p in normal if p in flagged], normal),
        precision=_rate([p for p in flagged_truth if truth[p]], flagged_truth),
        auroc=auroc(samples),
    )


def _rate(hits: Sequence[object], population: Sequence[object]) -> float:
    return len(hits) / len(population) if population else 0.0
