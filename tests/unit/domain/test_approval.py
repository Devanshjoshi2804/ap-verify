from __future__ import annotations

from tests.support import (
    PO_NUMBER,
    build_goods_receipt,
    build_invoice,
    build_purchase_order,
    build_raw_text,
)

from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.approval import (
    FinalDecision,
    approve,
    reconcile_with_anomaly,
    reconcile_with_duplicate,
    reconcile_with_vendor_risk,
)
from apverify.domain.checks import review
from apverify.domain.critique import ApprovalDecision
from apverify.domain.fraud import DuplicateMatch, DuplicateTier
from apverify.domain.matching import three_way_match
from apverify.domain.value_objects import Money
from apverify.domain.vendor_master import Severity, VendorRiskAssessment, VendorRiskKind


def _approved() -> FinalDecision:
    return FinalDecision(decision=ApprovalDecision.AUTO_APPROVE, reasons=("clean",))


def test_high_vendor_risk_holds_the_payment() -> None:
    assessment = VendorRiskAssessment(
        VendorRiskKind.BANK_CHANGE, Severity.HIGH, 1.0, "ACME", "bank changed"
    )
    result = reconcile_with_vendor_risk(_approved(), assessment)
    assert result.decision is ApprovalDecision.HOLD
    assert any("bank changed" in reason for reason in result.reasons)


def test_low_vendor_risk_adds_a_reason_but_does_not_change_the_decision() -> None:
    assessment = VendorRiskAssessment(VendorRiskKind.NEW_PAYEE, Severity.LOW, 0.2, "", "new vendor")
    result = reconcile_with_vendor_risk(_approved(), assessment)
    assert result.decision is ApprovalDecision.AUTO_APPROVE
    assert any("new vendor" in reason for reason in result.reasons)


def test_clean_vendor_risk_is_a_no_op() -> None:
    assessment = VendorRiskAssessment(VendorRiskKind.CLEAN, Severity.NONE, 1.0, "ACME", "known")
    result = reconcile_with_vendor_risk(_approved(), assessment)
    assert result == _approved()


def test_high_anomaly_holds_the_payment() -> None:
    assessment = AnomalyAssessment(0.95, AnomalySeverity.HIGH, "amount_spike", "11x median")
    result = reconcile_with_anomaly(_approved(), assessment)
    assert result.decision is ApprovalDecision.HOLD
    assert any("11x median" in reason for reason in result.reasons)


def test_medium_anomaly_routes_to_human_review() -> None:
    assessment = AnomalyAssessment(0.6, AnomalySeverity.MEDIUM, "amount_spike", "elevated")
    result = reconcile_with_anomaly(_approved(), assessment)
    assert result.decision is ApprovalDecision.HUMAN_REVIEW


def test_no_anomaly_is_a_no_op() -> None:
    assessment = AnomalyAssessment(0.1, AnomalySeverity.NONE, "amount_spike", "normal")
    result = reconcile_with_anomaly(_approved(), assessment)
    assert result == _approved()


def test_exact_duplicate_holds_the_payment() -> None:
    match = DuplicateMatch("ledger-1", DuplicateTier.EXACT_RESEND, 1.0, "identical to prior")
    result = reconcile_with_duplicate(_approved(), match)
    assert result.decision is ApprovalDecision.HOLD
    assert any("identical to prior" in reason for reason in result.reasons)


def test_near_duplicate_routes_to_human_review() -> None:
    match = DuplicateMatch("ledger-1", DuplicateTier.NEAR_DUPLICATE, 0.9, "near-duplicate")
    result = reconcile_with_duplicate(_approved(), match)
    assert result.decision is ApprovalDecision.HUMAN_REVIEW


def test_no_duplicate_is_a_no_op() -> None:
    assert reconcile_with_duplicate(_approved(), None) == _approved()


def _critic(invoice: object):  # type: ignore[no-untyped-def]
    return review(invoice, build_raw_text(invoice))  # type: ignore[arg-type]


def test_clean_extraction_and_clean_match_auto_approves() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    critic = _critic(invoice)
    match = three_way_match(invoice, build_purchase_order(), build_goods_receipt())

    decision = approve(critic, match)

    assert decision.decision is ApprovalDecision.AUTO_APPROVE


def test_untrusted_extraction_holds_even_with_a_clean_match() -> None:
    page_invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    hallucinated = build_invoice(purchase_order_ref=PO_NUMBER, total=Money.of("999999.00"))
    critic = review(hallucinated, build_raw_text(page_invoice))
    match = three_way_match(hallucinated, build_purchase_order(), build_goods_receipt())

    decision = approve(critic, match)

    assert decision.decision is ApprovalDecision.HOLD


def test_trusted_extraction_with_match_mismatch_routes_to_human() -> None:
    invoice = build_invoice(purchase_order_ref="PO-9999")
    critic = _critic(invoice)
    match = three_way_match(invoice, build_purchase_order(), build_goods_receipt())

    decision = approve(critic, match)

    assert decision.decision is ApprovalDecision.HUMAN_REVIEW
    assert any("match" in reason for reason in decision.reasons)


def test_review_grade_extraction_carries_its_reason_through() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER, currency="XYZ")
    critic = _critic(invoice)
    match = three_way_match(invoice, build_purchase_order(), build_goods_receipt())

    decision = approve(critic, match)

    assert decision.decision is ApprovalDecision.HUMAN_REVIEW
    assert any("confidence" in reason for reason in decision.reasons)


def test_no_purchase_order_routes_to_human() -> None:
    invoice = build_invoice()
    critic = _critic(invoice)
    match = three_way_match(invoice, None)

    decision = approve(critic, match)

    assert decision.decision is ApprovalDecision.HUMAN_REVIEW
