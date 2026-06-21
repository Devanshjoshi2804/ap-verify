"""Collusion detection from approval behaviour.

Collusion shows up not in invoice text but in *who approves what*: an approver who
funnels one vendor, clears amounts parked just under their own authorization limit, and
rubber-stamps them within seconds. This detects those patterns over a log of approval
records — cross-record and batch, not per-invoice. Pure domain logic; timestamps are
passed in as data, so there is no wall-clock here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from apverify.domain.value_objects import Money


class CollusionSeverity(Enum):
    NONE = "none"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    submitter: str
    approver: str
    vendor: str
    amount: Money
    submitted_at: datetime
    approved_at: datetime
    approver_limit: Money


@dataclass(frozen=True, slots=True)
class CollusionSignal:
    approver: str
    vendor: str
    concentration: float
    under_limit_rate: float
    rubber_stamp_rate: float
    score: float
    severity: CollusionSeverity
    reason: str


def detect_collusion(
    records: Sequence[ApprovalRecord],
    *,
    under_limit_band: float = 0.05,
    rubber_stamp_seconds: float = 60.0,
    flag_score: float = 0.6,
    min_approvals: int = 3,
) -> list[CollusionSignal]:
    approver_totals = Counter(record.approver for record in records)
    pairs: dict[tuple[str, str], list[ApprovalRecord]] = {}
    for record in records:
        pairs.setdefault((record.approver, record.vendor), []).append(record)

    signals: list[CollusionSignal] = []
    for (approver, vendor), group in pairs.items():
        if len(group) < min_approvals:
            continue
        concentration = len(group) / approver_totals[approver]
        under_limit_rate = _rate(
            group, lambda r: _just_under(r.amount, r.approver_limit, under_limit_band)
        )
        rubber_stamp_rate = _rate(
            group,
            lambda r: (r.approved_at - r.submitted_at).total_seconds() <= rubber_stamp_seconds,
        )
        score = (concentration + under_limit_rate + rubber_stamp_rate) / 3
        severity = (
            CollusionSeverity.HIGH
            if score >= 0.8
            else CollusionSeverity.MEDIUM
            if score >= flag_score
            else CollusionSeverity.NONE
        )
        signals.append(
            CollusionSignal(
                approver=approver,
                vendor=vendor,
                concentration=round(concentration, 4),
                under_limit_rate=round(under_limit_rate, 4),
                rubber_stamp_rate=round(rubber_stamp_rate, 4),
                score=round(score, 4),
                severity=severity,
                reason=(
                    f"approver {approver} cleared {concentration:.0%} of vendor {vendor}'s "
                    f"invoices, {under_limit_rate:.0%} just under limit, "
                    f"{rubber_stamp_rate:.0%} rubber-stamped"
                ),
            )
        )
    return sorted(signals, key=lambda signal: signal.score, reverse=True)


def _rate(group: Sequence[ApprovalRecord], predicate: Callable[[ApprovalRecord], bool]) -> float:
    return sum(1 for record in group if predicate(record)) / len(group)


def _just_under(amount: Money, limit: Money, band: float) -> bool:
    if limit.amount <= 0:
        return False
    gap = (limit.amount - amount.amount) / limit.amount
    return 0 <= gap <= band
