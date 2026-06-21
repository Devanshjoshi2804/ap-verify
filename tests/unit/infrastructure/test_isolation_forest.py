from __future__ import annotations

from decimal import Decimal

import pytest

pytest.importorskip("sklearn")

from apverify.domain.anomaly import AnomalySeverity
from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.value_objects import Money
from apverify.infrastructure.anomaly.isolation_forest import (
    IsolationForestDetector,
)


def _invoice(total: str) -> Invoice:
    amount = Money(Decimal(total))
    return Invoice(
        vendor_name="ACME",
        invoice_number="I",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=amount,
        tax=Money(Decimal("0")),
        total=amount,
        line_items=(LineItem("x", 1, amount, amount),),
        tax_breakdown=TaxBreakdown(),
        vendor_gstin="",
        purchase_order_ref="",
    )


def test_isolation_forest_scores_a_spike_above_a_normal() -> None:
    history = [_invoice(str(t)) for t in (90, 95, 100, 105, 110)]
    detector = IsolationForestDetector()
    spike = detector.score(_invoice("1000"), history)
    normal = detector.score(_invoice("102"), history)
    assert spike.score > normal.score


def test_insufficient_history_abstains() -> None:
    detector = IsolationForestDetector()
    result = detector.score(_invoice("1000"), [_invoice("100")])
    assert result.severity is AnomalySeverity.NONE
