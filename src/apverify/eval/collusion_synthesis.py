"""Synthesise a labelled approval log for the collusion benchmark.

Colluding pairs: one approver funnels one vendor — every approval just under the
approver's limit, approved within seconds. Normal pairs: an approver spreads work across
vendors, with varied amounts well below the limit and realistic (hours) approval latency.

Deterministic: timestamps and amounts derive from the index, no randomness.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from apverify.domain.collusion import ApprovalRecord
from apverify.domain.value_objects import Money

_BASE = datetime(2025, 6, 1, 9, 0, 0)
_LIMIT = Money(Decimal("50000"))


def build_collusion_log(
    pairs: int = 6, per_pair: int = 8
) -> tuple[list[ApprovalRecord], dict[tuple[str, str], bool]]:
    records: list[ApprovalRecord] = []
    truth: dict[tuple[str, str], bool] = {}
    for index in range(pairs):
        approver = f"approver{index}"
        if index % 2 == 0:
            vendor = f"vendor{index}"
            records.extend(_colluding(approver, vendor, per_pair))
            truth[(approver, vendor)] = True
        else:
            for variant in range(2):
                vendor = f"vendor{index}_{variant}"
                records.extend(_normal(approver, vendor, per_pair))
                truth[(approver, vendor)] = False
    return records, truth


def _colluding(approver: str, vendor: str, count: int) -> list[ApprovalRecord]:
    return [
        ApprovalRecord(
            submitter="submitter",
            approver=approver,
            vendor=vendor,
            amount=Money(Decimal("49500")),  # just under the 50000 limit
            submitted_at=_BASE + timedelta(days=day),
            approved_at=_BASE + timedelta(days=day, seconds=10),  # rubber-stamped
            approver_limit=_LIMIT,
        )
        for day in range(count)
    ]


def _normal(approver: str, vendor: str, count: int) -> list[ApprovalRecord]:
    return [
        ApprovalRecord(
            submitter="submitter",
            approver=approver,
            vendor=vendor,
            amount=Money(Decimal(str(10000 + day * 2000))),  # varied, well below limit
            submitted_at=_BASE + timedelta(days=day),
            approved_at=_BASE + timedelta(days=day, hours=5),  # real review latency
            approver_limit=_LIMIT,
        )
        for day in range(count)
    ]
