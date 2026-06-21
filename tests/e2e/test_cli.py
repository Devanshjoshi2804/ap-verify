from __future__ import annotations

from pathlib import Path

import pytest
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
from typer.testing import CliRunner

from apverify.application.process_invoice import ProcessInvoiceUseCase
from apverify.application.review_payable import ReviewPayableUseCase
from apverify.domain.value_objects import Money
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository
from apverify.interface.cli import main as cli_main

_RUNNER = CliRunner()
_SAMPLE = Path(__file__).resolve().parents[2] / "samples" / "clean_invoice_01.pdf"


def _wire(monkeypatch: pytest.MonkeyPatch, use_case: ProcessInvoiceUseCase) -> None:
    """Replace the composition root so the CLI runs without network or binaries."""
    monkeypatch.setattr(cli_main, "build_use_case", lambda *args, **kwargs: use_case)


def test_cli_auto_approves_a_clean_invoice(monkeypatch: pytest.MonkeyPatch) -> None:
    invoice = build_invoice()
    use_case = ProcessInvoiceUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(invoice),
        ocr=FakeOcr(build_raw_text(invoice)),
    )
    _wire(monkeypatch, use_case)

    result = _RUNNER.invoke(cli_main.app, ["run", str(_SAMPLE)])

    assert result.exit_code == 0
    assert "AUTO_APPROVE" in result.stdout


def test_cli_holds_and_reports_the_flag_for_a_hallucinated_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_case = ProcessInvoiceUseCase(
        renderer=FakeRenderer(),
        extractor=FakeExtractor(build_invoice(total=Money.of("999999.00"))),
        ocr=FakeOcr(build_raw_text(build_invoice())),
    )
    _wire(monkeypatch, use_case)

    result = _RUNNER.invoke(cli_main.app, ["run", str(_SAMPLE)])

    assert result.exit_code == 1
    assert "HOLD" in result.stdout
    assert "does not appear" in result.stdout


def test_review_command_runs_the_full_pipeline_and_auto_approves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(cli_main, "build_review_use_case", lambda *a, **k: use_case)

    result = _RUNNER.invoke(cli_main.app, ["review", str(_SAMPLE)])

    assert result.exit_code == 0
    assert "AUTO_APPROVE" in result.stdout
    assert "match" in result.stdout.lower()
