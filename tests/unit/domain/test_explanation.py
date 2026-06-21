from __future__ import annotations

from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.explanation import (
    Factor,
    explain_anomaly,
    explain_duplicate,
    explain_vendor_risk,
    explanation,
)
from apverify.domain.fraud import DuplicateMatch, DuplicateTier
from apverify.domain.vendor_master import Severity, VendorRiskAssessment, VendorRiskKind


def test_factors_are_ranked_by_absolute_contribution() -> None:
    result = explanation(
        "test",
        "headline",
        [Factor("a", "1", 0.2, ""), Factor("b", "2", -0.9, ""), Factor("c", "3", 0.5, "")],
    )
    assert [factor.signal for factor in result.factors] == ["b", "c", "a"]


def test_vendor_risk_explanation_leads_with_the_kind() -> None:
    assessment = VendorRiskAssessment(
        VendorRiskKind.BANK_CHANGE, Severity.HIGH, 1.0, "ACME", "bank changed"
    )
    result = explain_vendor_risk(assessment)
    assert result.factors[0].value == "bank_change"
    assert "bank changed" in result.factors[0].detail


def test_anomaly_explanation_names_the_top_feature() -> None:
    assessment = AnomalyAssessment(0.95, AnomalySeverity.HIGH, "amount_spike", "11x median")
    result = explain_anomaly(assessment)
    assert result.factors[0].signal == "amount_spike"
    assert result.factors[0].contribution == 0.95


def test_duplicate_explanation_leads_with_the_tier() -> None:
    match = DuplicateMatch("ledger-1", DuplicateTier.EXACT_RESEND, 1.0, "identical to prior")
    result = explain_duplicate(match)
    assert result.factors[0].signal == "duplicate_tier"
    assert result.factors[0].value == "exact_resend"
