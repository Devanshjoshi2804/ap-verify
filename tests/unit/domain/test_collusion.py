from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from apverify.domain.collusion import (
    ApprovalRecord,
    CollusionSeverity,
    detect_collusion,
)
from apverify.domain.value_objects import Money

_BASE = datetime(2025, 6, 1, 9, 0, 0)
_LIMIT = Money(Decimal("50000"))


def _record(
    approver: str, vendor: str, amount: str, *, day: int, latency_s: float
) -> ApprovalRecord:
    submitted = _BASE + timedelta(days=day)
    return ApprovalRecord(
        submitter="bob",
        approver=approver,
        vendor=vendor,
        amount=Money(Decimal(amount)),
        submitted_at=submitted,
        approved_at=submitted + timedelta(seconds=latency_s),
        approver_limit=_LIMIT,
    )


def _colluding(approver: str, vendor: str, count: int) -> list[ApprovalRecord]:
    return [_record(approver, vendor, "49500", day=j, latency_s=10) for j in range(count)]


def test_a_funneled_just_under_rubber_stamped_pair_is_high() -> None:
    signals = detect_collusion(_colluding("alice", "acme", 5))
    assert signals[0].severity is CollusionSeverity.HIGH
    assert signals[0].approver == "alice"
    assert signals[0].concentration == 1.0


def test_a_diverse_normal_approver_is_not_flagged() -> None:
    records = [
        _record("carol", f"vendor{j % 2}", str(10000 + j * 2000), day=j, latency_s=18000)
        for j in range(6)
    ]
    signals = detect_collusion(records)
    assert all(s.severity is CollusionSeverity.NONE for s in signals)


def test_a_pair_below_the_minimum_is_not_scored() -> None:
    signals = detect_collusion(_colluding("alice", "acme", 2))  # only 2 < min 3
    assert signals == []


def test_zero_limit_is_not_treated_as_under_limit() -> None:
    submitted = _BASE
    records = [
        ApprovalRecord(
            submitter="bob",
            approver="dan",
            vendor="acme",
            amount=Money(Decimal("49500")),
            submitted_at=submitted + timedelta(days=j),
            approved_at=submitted + timedelta(days=j, seconds=10),
            approver_limit=Money(Decimal("0")),
        )
        for j in range(3)
    ]
    signals = detect_collusion(records)
    assert signals[0].under_limit_rate == 0.0


def test_reason_names_the_patterns() -> None:
    signals = detect_collusion(_colluding("alice", "acme", 5))
    assert "alice" in signals[0].reason and "acme" in signals[0].reason
