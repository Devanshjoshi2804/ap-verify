"""Shared machinery for evaluating the critic over real-world datasets.

A dataset adapter (CORD, DocILE, …) maps its records into a ``DatasetExample`` —
a domain ``Invoice`` plus the OCR ``RawText`` a faithful reader would produce.
This module owns the amount parsing they share, the example type, and the run loop
that reports the auto-approve rate and a breakdown of why receipts are held.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from apverify.domain.checks import review
from apverify.domain.critique import DEFAULT_POLICY, ApprovalDecision, Policy
from apverify.domain.errors import InvalidMoneyError
from apverify.domain.invoice import Invoice
from apverify.domain.ocr import RawText
from apverify.domain.value_objects import Money

_NON_AMOUNT = re.compile(r"[^\d.,]")


@dataclass(frozen=True, slots=True)
class DatasetExample:
    label: str
    invoice: Invoice
    raw_text: RawText


@dataclass(frozen=True, slots=True)
class DatasetReport:
    total: int
    decisions: dict[str, int]
    failed_checks: dict[str, int]

    @property
    def auto_approve_rate(self) -> float:
        if not self.total:
            return 0.0
        return self.decisions.get(ApprovalDecision.AUTO_APPROVE.value, 0) / self.total


def run_dataset_eval(
    examples: Iterable[DatasetExample], policy: Policy = DEFAULT_POLICY
) -> DatasetReport:
    decisions: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    total = 0
    for example in examples:
        total += 1
        report = review(example.invoice, example.raw_text, policy)
        decisions[report.decision.value] += 1
        for flag in report.flags:
            failed[f"{flag.field}/{flag.category}"] += 1
    return DatasetReport(total=total, decisions=dict(decisions), failed_checks=dict(failed))


def parse_amount(raw: str) -> Money:
    """Parse a printed amount tolerant of mixed thousands / decimal conventions."""
    digits = _NON_AMOUNT.sub("", raw)
    if not digits:
        raise ValueError(f"no digits in amount {raw!r}")
    try:
        return Money.of(_normalise_separators(digits))
    except InvalidMoneyError as exc:
        raise ValueError(f"cannot parse amount {raw!r}") from exc


def _normalise_separators(value: str) -> str:
    if "." in value and "," in value:
        if value.rfind(",") > value.rfind("."):
            return value.replace(".", "").replace(",", ".")
        return value.replace(",", "")
    if value.count(",") == 1 and len(value.split(",")[1]) in (1, 2):
        return value.replace(",", ".")
    if value.count(".") == 1 and len(value.split(".")[1]) in (1, 2):
        return value
    return value.replace(",", "").replace(".", "")
