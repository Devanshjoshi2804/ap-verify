from __future__ import annotations

from decimal import Decimal

from apverify.domain.anomaly import (
    AnomalySeverity,
    RobustAnomalyDetector,
    extract_features,
)
from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.value_objects import Money


def _invoice(total: str) -> Invoice:
    amount = Money(Decimal(total))
    return Invoice(
        vendor_name="ACME Steel Pvt Ltd",
        invoice_number="INV-1",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=amount,
        tax=Money(Decimal("0")),
        total=amount,
        line_items=(LineItem("Widget", 1, amount, amount),),
        tax_breakdown=TaxBreakdown(),
        vendor_gstin="",
        purchase_order_ref="",
    )


def _history(totals: list[str]) -> list[Invoice]:
    return [_invoice(total) for total in totals]


_NORMAL_HISTORY = _history(["90", "95", "100", "105", "110"])
_DETECTOR = RobustAnomalyDetector()


def test_in_range_amount_is_not_anomalous() -> None:
    result = _DETECTOR.score(_invoice("102"), _NORMAL_HISTORY)
    assert result.severity is AnomalySeverity.NONE


def test_amount_spike_is_high_severity() -> None:
    result = _DETECTOR.score(_invoice("1000"), _NORMAL_HISTORY)
    assert result.severity is AnomalySeverity.HIGH
    assert result.top_feature == "amount_spike"
    assert "median" in result.reason


def test_amount_just_under_a_round_limit_is_flagged_for_gaming() -> None:
    history = _history(["9000", "9100", "9200", "9300", "9150"])
    result = _DETECTOR.score(_invoice("9950"), history)
    assert result.severity is AnomalySeverity.HIGH
    assert result.top_feature == "threshold_gaming"
    assert "approval limit" in result.reason


def test_insufficient_history_abstains() -> None:
    result = _DETECTOR.score(_invoice("1000"), _history(["100", "100"]))
    assert result.severity is AnomalySeverity.NONE
    assert result.score == 0.0
    assert "insufficient history" in result.reason


def test_identical_history_does_not_divide_by_zero() -> None:
    result = _DETECTOR.score(_invoice("100"), _history(["100", "100", "100", "100"]))
    assert result.severity is AnomalySeverity.NONE  # same as history, no anomaly


def test_extract_features_with_no_history_reports_zero_z() -> None:
    features = extract_features(_invoice("100"), [])
    assert features.amount_robust_z == 0.0
    assert features.history_size == 0


def test_zero_amount_has_no_threshold_proximity() -> None:
    features = extract_features(_invoice("0"), [])
    assert features.threshold_proximity == 0.0
