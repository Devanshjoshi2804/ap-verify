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
from apverify.domain.anomaly import RobustAnomalyDetector
from apverify.domain.critique import ApprovalDecision
from apverify.domain.value_objects import Money
from apverify.infrastructure.anomaly.history import InMemoryVendorHistory
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository

_DOC = Path("ignored-by-fakes.pdf")
# The candidate is the consistent default invoice (total 184,200) that the critic and
# 3-way match both pass; only the vendor's *history* differs between the two cases, so
# the anomaly step is the sole driver of the decision change.
_INVOICE = build_invoice(purchase_order_ref=PO_NUMBER)


def _history(totals: list[int]) -> InMemoryVendorHistory:
    return InMemoryVendorHistory([build_invoice(total=Money.of(str(total))) for total in totals])


def _use_case(history: InMemoryVendorHistory) -> ReviewPayableUseCase:
    return ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(_INVOICE),
        ocr=FakeOcr(build_raw_text(_INVOICE)),
        procurement=InMemoryProcurementRepository(
            purchase_orders=[build_purchase_order()],
            goods_receipts=[build_goods_receipt()],
        ),
        anomaly_detector=RobustAnomalyDetector(),
        vendor_history=history,
    )


def test_amount_spike_holds_the_payment() -> None:
    # History clustered far below the candidate's 184,200 makes it a spike.
    review = _use_case(_history([9000, 9100, 9200, 9300, 9150])).execute(_DOC)
    assert review.decision.decision is ApprovalDecision.HOLD
    assert review.anomaly is not None
    assert any("anomaly" in reason for reason in review.decision.reasons)


def test_in_range_amount_stays_auto_approved() -> None:
    # History centred on the candidate's amount: nothing unusual.
    review = _use_case(_history([180000, 184000, 184200, 185000, 188000])).execute(_DOC)
    assert review.decision.decision is ApprovalDecision.AUTO_APPROVE
    assert review.anomaly is not None


def test_flagged_invoice_carries_a_structured_explanation() -> None:
    review = _use_case(_history([9000, 9100, 9200, 9300, 9150])).execute(_DOC)
    assert review.explanations  # at least one explanation for the anomaly flag
    assert any(e.source == "anomaly" for e in review.explanations)


def test_clean_invoice_has_no_explanations() -> None:
    review = _use_case(_history([180000, 184000, 184200, 185000, 188000])).execute(_DOC)
    assert review.explanations == ()
