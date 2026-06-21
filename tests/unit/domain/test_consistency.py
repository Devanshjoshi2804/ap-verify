from __future__ import annotations

from tests.support import build_invoice

from apverify.domain.approval import FinalDecision, reconcile_with_consistency
from apverify.domain.consistency import Agreement, compare_extractions
from apverify.domain.critique import ApprovalDecision, InvoiceField
from apverify.domain.value_objects import Money


def test_identical_extractions_fully_agree() -> None:
    invoice = build_invoice()
    report = compare_extractions(invoice, invoice)

    assert report.disagreements == ()
    assert all(c.agreement is Agreement.AGREES for c in report.comparisons)


def test_differing_total_is_a_disagreement() -> None:
    report = compare_extractions(build_invoice(), build_invoice(total=Money.of("184000.00")))

    assert {c.field for c in report.disagreements} == {InvoiceField.TOTAL}


def test_differing_vendor_is_a_disagreement() -> None:
    report = compare_extractions(build_invoice(), build_invoice(vendor_name="Phantom Traders"))

    assert any(c.field is InvoiceField.VENDOR for c in report.disagreements)


def test_minor_amount_difference_within_tolerance_agrees() -> None:
    report = compare_extractions(build_invoice(), build_invoice(total=Money.of("184200.03")))

    assert report.disagreements == ()


def test_disagreement_on_a_critical_field_holds_the_payment() -> None:
    base = FinalDecision(ApprovalDecision.AUTO_APPROVE, ("clean",))
    report = compare_extractions(build_invoice(), build_invoice(total=Money.of("184000.00")))

    result = reconcile_with_consistency(base, report)

    assert result.decision is ApprovalDecision.HOLD
    assert any("disagree" in reason for reason in result.reasons)


def test_agreement_leaves_the_decision_untouched() -> None:
    base = FinalDecision(ApprovalDecision.AUTO_APPROVE, ("clean",))
    invoice = build_invoice()

    result = reconcile_with_consistency(base, compare_extractions(invoice, invoice))

    assert result is base
