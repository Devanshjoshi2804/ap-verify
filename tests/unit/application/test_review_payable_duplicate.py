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
from apverify.domain.fraud import IdentifiedInvoice
from apverify.infrastructure.invoice_ledger import InMemoryInvoiceLedger
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository

_DOC = Path("ignored-by-fakes.pdf")
_INVOICE = build_invoice(purchase_order_ref=PO_NUMBER)


def _use_case(ledger: InMemoryInvoiceLedger | None) -> ReviewPayableUseCase:
    return ReviewPayableUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(_INVOICE),
        ocr=FakeOcr(build_raw_text(_INVOICE)),
        procurement=InMemoryProcurementRepository(
            purchase_orders=[build_purchase_order()],
            goods_receipts=[build_goods_receipt()],
        ),
        invoice_ledger=ledger,
    )


def test_resend_of_a_ledger_invoice_holds_the_payment() -> None:
    # The same invoice is already on the ledger — a verbatim resend.
    ledger = InMemoryInvoiceLedger([IdentifiedInvoice("ledger-1", _INVOICE)])
    review = _use_case(ledger).execute(_DOC)
    assert review.decision.decision is ApprovalDecision.HOLD
    assert review.duplicate is not None
    assert any("duplicate" in reason for reason in review.decision.reasons)
    assert any(e.source == "duplicate" for e in review.explanations)


def test_no_prior_means_no_duplicate() -> None:
    review = _use_case(InMemoryInvoiceLedger([])).execute(_DOC)
    assert review.decision.decision is ApprovalDecision.AUTO_APPROVE
    assert review.duplicate is None
