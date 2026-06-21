from __future__ import annotations

from pathlib import Path

from tests.support import (
    PO_NUMBER,
    FakeAuditor,
    FakeExtractor,
    FakeOcr,
    FakeRenderer,
    build_goods_receipt,
    build_invoice,
    build_purchase_order,
    build_raw_text,
)

from apverify.application.review_payable import ReviewPayableUseCase
from apverify.domain.audit import AuditVerdict
from apverify.domain.critique import ApprovalDecision, InvoiceField
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository

_DOC = Path("ignored-by-fakes.pdf")


def _use_case(auditor: FakeAuditor) -> ReviewPayableUseCase:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    return ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(invoice),
        ocr=FakeOcr(build_raw_text(invoice)),
        procurement=InMemoryProcurementRepository(
            purchase_orders=[build_purchase_order()],
            goods_receipts=[build_goods_receipt()],
        ),
        auditor=auditor,
        audit_below=1.1,  # force every field through the auditor for the test
    )


def test_auditor_can_veto_an_otherwise_clean_auto_approval() -> None:
    auditor = FakeAuditor(
        [
            AuditVerdict(
                InvoiceField.TOTAL, trustworthy=False, confidence=0.8, reason="off by a lakh"
            )
        ]
    )
    review = _use_case(auditor).execute(_DOC)

    assert review.decision.decision is ApprovalDecision.HOLD
    assert any(entry.step == "audit" for entry in review.trace)
    assert review.audit_verdicts


def test_trustworthy_audit_leaves_auto_approval_intact() -> None:
    auditor = FakeAuditor(
        [AuditVerdict(InvoiceField.TOTAL, trustworthy=True, confidence=0.99, reason="ok")]
    )
    review = _use_case(auditor).execute(_DOC)

    assert review.decision.decision is ApprovalDecision.AUTO_APPROVE


def test_auditor_is_skipped_when_no_field_is_in_doubt() -> None:
    auditor = FakeAuditor([])
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    use_case = ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(invoice),
        ocr=FakeOcr(build_raw_text(invoice)),
        procurement=InMemoryProcurementRepository(
            purchase_orders=[build_purchase_order()],
            goods_receipts=[build_goods_receipt()],
        ),
        auditor=auditor,  # default threshold: a clean invoice has nothing below it
    )
    review = use_case.execute(_DOC)

    assert auditor.audited_fields == ()
    assert review.decision.decision is ApprovalDecision.AUTO_APPROVE
    assert all(entry.step != "audit" for entry in review.trace)


def test_no_auditor_means_no_audit_step() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    use_case = ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(invoice),
        ocr=FakeOcr(build_raw_text(invoice)),
        procurement=InMemoryProcurementRepository(
            purchase_orders=[build_purchase_order()],
            goods_receipts=[build_goods_receipt()],
        ),
    )
    review = use_case.execute(_DOC)

    assert review.audit_verdicts == ()
    assert all(entry.step != "audit" for entry in review.trace)
