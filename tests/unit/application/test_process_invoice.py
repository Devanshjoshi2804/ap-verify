from __future__ import annotations

from pathlib import Path

from tests.support import FakeExtractor, FakeOcr, FakeRenderer, build_invoice, build_raw_text

from apverify.application.process_invoice import ProcessInvoiceUseCase
from apverify.domain.critique import ApprovalDecision
from apverify.domain.value_objects import Money


def test_use_case_returns_an_approved_review_when_everything_reconciles() -> None:
    invoice = build_invoice()
    use_case = ProcessInvoiceUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(invoice),
        ocr=FakeOcr(build_raw_text(invoice)),
    )

    review = use_case.execute(Path("ignored-by-fake.pdf"))

    assert review.invoice is invoice
    assert review.report.decision is ApprovalDecision.AUTO_APPROVE


def test_use_case_surfaces_a_hold_when_the_critic_rejects_extraction() -> None:
    page_invoice = build_invoice()
    hallucinated = build_invoice(total=Money.of("999999.00"))
    use_case = ProcessInvoiceUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(hallucinated),
        ocr=FakeOcr(build_raw_text(page_invoice)),
    )

    review = use_case.execute(Path("ignored-by-fake.pdf"))

    assert review.report.decision is ApprovalDecision.HOLD
