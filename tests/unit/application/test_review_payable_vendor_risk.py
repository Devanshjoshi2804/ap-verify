from __future__ import annotations

from pathlib import Path

from tests.support import (
    PO_NUMBER,
    FakeExtractor,
    FakeOcr,
    FakeRenderer,
    build_goods_receipt,
    build_invoice,
    build_purchase_order,
    build_raw_text,
)

from apverify.application.review_payable import ReviewPayableUseCase
from apverify.domain.critique import ApprovalDecision
from apverify.domain.vendor_master import KnownVendor
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository

_DOC = Path("ignored-by-fakes.pdf")
_MASTER = (KnownVendor("ACME Steel Pvt Ltd", frozenset({"ACCT-0001"})),)


class _Master:
    def known_vendors(self) -> tuple[KnownVendor, ...]:
        return _MASTER


def _use_case(invoice: object) -> ReviewPayableUseCase:
    return ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(invoice),  # type: ignore[arg-type]
        ocr=FakeOcr(build_raw_text(invoice)),  # type: ignore[arg-type]
        procurement=InMemoryProcurementRepository(
            purchase_orders=[build_purchase_order()],
            goods_receipts=[build_goods_receipt()],
        ),
        vendor_master=_Master(),
    )


def test_changed_bank_account_holds_the_payment() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER, bank_account="ACCT-9999")
    review = _use_case(invoice).execute(_DOC)

    assert review.decision.decision is ApprovalDecision.HOLD
    assert review.vendor_risk is not None
    assert review.vendor_risk.kind.value == "bank_change"
    assert any("vendor-risk" in reason for reason in review.decision.reasons)
    assert any(entry.step == "vendor-risk" for entry in review.trace)


def test_known_vendor_known_bank_stays_auto_approved() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER, bank_account="ACCT-0001")
    review = _use_case(invoice).execute(_DOC)

    assert review.decision.decision is ApprovalDecision.AUTO_APPROVE
    assert review.vendor_risk is not None
    assert review.vendor_risk.kind.value == "clean"
