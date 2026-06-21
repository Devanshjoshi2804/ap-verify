"""Cross-provider extraction leaderboard.

The accuracy harness scores one extractor; this ranks *several* over the same real
invoices so the trade-off between providers is explicit — the highest-quality model,
the best free/local option, and everything between. Ranking is pure and unit-tested;
the live multi-provider run lives in the leaderboard runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from apverify.eval.accuracy import AccuracyReport


@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    provider: str
    macro_f1: float
    line_item_f1: float | None  # None when the dataset carries no line-item ground truth
    documents: int


def rank_providers(reports: Mapping[str, AccuracyReport]) -> tuple[LeaderboardRow, ...]:
    """One row per provider, ranked by per-field macro-F1 (ties broken by name)."""
    rows = [
        LeaderboardRow(
            provider=provider,
            macro_f1=report.macro_f1,
            line_item_f1=report.line_items.f1 if report.line_items is not None else None,
            documents=report.documents,
        )
        for provider, report in reports.items()
    ]
    return tuple(sorted(rows, key=lambda row: (-row.macro_f1, row.provider)))
