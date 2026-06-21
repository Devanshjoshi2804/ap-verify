from __future__ import annotations

from apverify.domain.approval import FinalDecision, reconcile_with_audit
from apverify.domain.audit import AuditVerdict
from apverify.domain.critique import ApprovalDecision, InvoiceField

_AUTO = FinalDecision(ApprovalDecision.AUTO_APPROVE, ("clean",))


def _verdict(field: InvoiceField, *, trustworthy: bool) -> AuditVerdict:
    return AuditVerdict(field=field, trustworthy=trustworthy, confidence=0.9, reason="because")


def test_trustworthy_verdicts_leave_the_decision_untouched() -> None:
    result = reconcile_with_audit(_AUTO, [_verdict(InvoiceField.TOTAL, trustworthy=True)])
    assert result is _AUTO


def test_distrusted_critical_field_holds_the_payment() -> None:
    result = reconcile_with_audit(_AUTO, [_verdict(InvoiceField.TOTAL, trustworthy=False)])
    assert result.decision is ApprovalDecision.HOLD
    assert any("auditor distrusts" in reason for reason in result.reasons)


def test_distrusted_non_critical_field_routes_to_human() -> None:
    result = reconcile_with_audit(_AUTO, [_verdict(InvoiceField.INVOICE_NUMBER, trustworthy=False)])
    assert result.decision is ApprovalDecision.HUMAN_REVIEW


def test_audit_never_upgrades_a_held_decision() -> None:
    held = FinalDecision(ApprovalDecision.HOLD, ("critic hold",))
    result = reconcile_with_audit(held, [_verdict(InvoiceField.CURRENCY, trustworthy=False)])
    assert result.decision is ApprovalDecision.HOLD
