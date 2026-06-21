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
from apverify.domain.value_objects import Money
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository

_DOC = Path("ignored-by-fakes.pdf")


def _use_case(invoice: object, repository: InMemoryProcurementRepository) -> ReviewPayableUseCase:
    return ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(invoice),  # type: ignore[arg-type]
        ocr=FakeOcr(build_raw_text(invoice)),  # type: ignore[arg-type]
        procurement=repository,
    )


def _seeded_repository() -> InMemoryProcurementRepository:
    return InMemoryProcurementRepository(
        purchase_orders=[build_purchase_order()],
        goods_receipts=[build_goods_receipt()],
    )


def test_full_pipeline_auto_approves_a_clean_matched_invoice() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER)
    review = _use_case(invoice, _seeded_repository()).execute(_DOC)

    assert review.decision.decision is ApprovalDecision.AUTO_APPROVE
    assert [entry.step for entry in review.trace] == [
        "render",
        "extract",
        "ocr",
        "critic",
        "match",
        "approve",
    ]


def test_untrusted_extraction_holds_regardless_of_match() -> None:
    invoice = build_invoice(purchase_order_ref=PO_NUMBER, total=Money.of("999999.00"))
    review = _use_case(invoice, _seeded_repository()).execute(_DOC)

    assert review.decision.decision is ApprovalDecision.HOLD


def test_missing_purchase_order_routes_to_human_review() -> None:
    invoice = build_invoice(purchase_order_ref="PO-UNKNOWN")
    review = _use_case(invoice, _seeded_repository()).execute(_DOC)

    assert review.decision.decision is ApprovalDecision.HUMAN_REVIEW
