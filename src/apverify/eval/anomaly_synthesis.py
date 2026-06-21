"""Synthesise a labelled anomaly benchmark over base invoices.

For each base vendor we build a plausible history clustered around a median, then emit
the anomalies (an amount spike far above the median; an amount parked just under a round
approval limit) and the hard negative (an amount within the vendor's usual spread). The
normal case is what makes the false-positive number mean something.

Deterministic: fixed transforms of the base, no randomness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal

from apverify.domain.anomaly import _round_above
from apverify.domain.invoice import Invoice
from apverify.domain.value_objects import Money
from apverify.eval.synthetic import GroundTruth

AMOUNT_SPIKE = "amount_spike"
THRESHOLD_GAMING = "threshold_gaming"
NORMAL = "normal"

SCENARIOS = (AMOUNT_SPIKE, THRESHOLD_GAMING, NORMAL)
ANOMALY_KINDS = (AMOUNT_SPIKE, THRESHOLD_GAMING)

# A deterministic ±spread around the vendor's median, in multiples of the base total.
_HISTORY_SPREAD = (
    Decimal("0.90"),
    Decimal("0.95"),
    Decimal("1.00"),
    Decimal("1.05"),
    Decimal("1.10"),
)


@dataclass(frozen=True, slots=True)
class AnomalyCase:
    invoice: Invoice
    history: tuple[Invoice, ...]
    is_anomaly: bool
    kind: str


def build_anomaly_cases(base: Sequence[GroundTruth]) -> list[AnomalyCase]:
    cases: list[AnomalyCase] = []
    for truth in base:
        invoice = truth.invoice
        median = invoice.total.amount
        history = tuple(_at(invoice, median * factor) for factor in _HISTORY_SPREAD)
        cases.extend(
            [
                AnomalyCase(_at(invoice, median * 10), history, True, AMOUNT_SPIKE),
                AnomalyCase(
                    _at(invoice, _just_under_round(median)), history, True, THRESHOLD_GAMING
                ),
                AnomalyCase(_at(invoice, median * Decimal("1.02")), history, False, NORMAL),
            ]
        )
    return cases


def _at(invoice: Invoice, total: Decimal) -> Invoice:
    money = Money(total)
    return replace(invoice, total=money, subtotal=money)


def _just_under_round(median: Decimal) -> Decimal:
    """A value 0.5% below the next round number above the median — plausible for the
    vendor (small robust-z) yet parked under an approval limit."""
    limit = Decimal(str(_round_above(float(median))))
    return (limit * Decimal("0.995")).quantize(Decimal("1"))
