from __future__ import annotations

from pathlib import Path

from tests.support import (
    PO_NUMBER,
    FailingExtractor,
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
from apverify.domain.value_objects import Money
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository

_DOC = Path("ignored-by-fakes.pdf")


def _use_case(primary: object, secondary: object) -> ReviewPayableUseCase:
    return ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(primary),  # type: ignore[arg-type]
        ocr=FakeOcr(build_raw_text(primary)),  # type: ignore[arg-type]
        procurement=InMemoryProcurementRepository(
            purchase_orders=[build_purchase_order()],
            goods_receipts=[build_goods_receipt()],
        ),
        secondary_extractor=FakeExtractor(secondary),  # type: ignore[arg-type]
    )


def test_agreeing_second_extraction_keeps_auto_approval() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    review = _use_case(invoice, invoice).execute(_DOC)

    assert review.decision.decision is ApprovalDecision.AUTO_APPROVE
    assert any(entry.step == "consistency" for entry in review.trace)
    assert review.consistency_report is not None
    assert review.consistency_report.disagreements == ()


def test_disagreeing_second_extraction_holds_on_the_total() -> None:
    primary = build_invoice(purchase_order_ref=PO_NUMBER)
    secondary = build_invoice(purchase_order_ref=PO_NUMBER, total=Money.of("184000.00"))
    review = _use_case(primary, secondary).execute(_DOC)

    assert review.decision.decision is ApprovalDecision.HOLD
    assert review.consistency_report is not None
    assert review.consistency_report.disagreements


def test_secondary_extractor_failure_degrades_gracefully() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    use_case = ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(invoice),
        ocr=FakeOcr(build_raw_text(invoice)),
        procurement=InMemoryProcurementRepository(
            purchase_orders=[build_purchase_order()],
            goods_receipts=[build_goods_receipt()],
        ),
        secondary_extractor=FailingExtractor(),
    )
    review = use_case.execute(_DOC)

    assert review.consistency_report is None
    assert review.decision.decision is ApprovalDecision.AUTO_APPROVE
    assert any(e.step == "consistency" and "unavailable" in e.detail for e in review.trace)
