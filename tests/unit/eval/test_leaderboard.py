from __future__ import annotations

from apverify.eval.accuracy import AccuracyReport, FieldStats, LineItemStats
from apverify.eval.leaderboard import LeaderboardRow, rank_providers


def _report(matched: int, mismatched: int, lines: LineItemStats | None = None) -> AccuracyReport:
    return AccuracyReport(
        documents=1,
        stats=(FieldStats("vendor", matched, mismatched, 0),),
        line_items=lines,
    )


def test_ranks_providers_by_macro_f1_descending() -> None:
    reports = {
        "weak": _report(1, 1),  # f1 0.5
        "strong": _report(1, 0),  # f1 1.0
    }
    ranked = rank_providers(reports)
    assert [row.provider for row in ranked] == ["strong", "weak"]
    assert isinstance(ranked[0], LeaderboardRow)
    assert ranked[0].macro_f1 == 1.0


def test_row_carries_line_item_f1_when_present() -> None:
    reports = {"p": _report(1, 0, lines=LineItemStats(matched=1, spurious=0, missed=1))}
    row = rank_providers(reports)[0]
    assert row.line_item_f1 is not None


def test_line_item_f1_is_none_when_absent() -> None:
    row = rank_providers({"p": _report(1, 0)})[0]
    assert row.line_item_f1 is None
