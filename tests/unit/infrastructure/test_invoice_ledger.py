from __future__ import annotations

import json
from pathlib import Path

import pytest

from apverify.application.ports import InvoiceLedger
from apverify.infrastructure.invoice_ledger import (
    InvoiceLedgerError,
    load_invoice_ledger,
)


def test_load_invoice_ledger_parses_and_identifies_by_number(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "invoices": [
                    {
                        "vendor_name": "ACME",
                        "invoice_number": "INV-1",
                        "invoice_date": "04-06-2025",
                        "currency": "INR",
                        "subtotal": "100",
                        "tax": "0",
                        "total": "100",
                        "line_items": [],
                    }
                ]
            }
        )
    )
    ledger = load_invoice_ledger(path)
    assert isinstance(ledger, InvoiceLedger)
    known = ledger.known_invoices()
    assert known[0].identifier == "INV-1"
    assert known[0].invoice.vendor_name == "ACME"


def test_load_invoice_ledger_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InvoiceLedgerError):
        load_invoice_ledger(tmp_path / "nope.json")
