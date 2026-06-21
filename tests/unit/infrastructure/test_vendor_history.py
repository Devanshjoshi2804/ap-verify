from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from apverify.application.ports import VendorHistoryRepository
from apverify.domain.invoice import Invoice, LineItem, TaxBreakdown
from apverify.domain.value_objects import Money
from apverify.infrastructure.anomaly.history import (
    InMemoryVendorHistory,
    VendorHistoryError,
    load_vendor_history,
)


def _invoice(vendor: str, total: str) -> Invoice:
    amount = Money(Decimal(total))
    return Invoice(
        vendor_name=vendor,
        invoice_number="H",
        invoice_date="04-06-2025",
        currency="INR",
        subtotal=amount,
        tax=Money(Decimal("0")),
        total=amount,
        line_items=(LineItem("x", 1, amount, amount),),
        tax_breakdown=TaxBreakdown(),
        vendor_gstin="",
        purchase_order_ref="",
    )


def test_history_is_grouped_by_vendor() -> None:
    history = InMemoryVendorHistory(
        [_invoice("ACME", "100"), _invoice("ACME", "110"), _invoice("Other", "5")]
    )
    assert isinstance(history, VendorHistoryRepository)
    matched = history.history_for(_invoice("ACME", "999"))
    assert {str(inv.total.amount) for inv in matched} == {"100.00", "110.00"}


def test_load_vendor_history_parses_a_file(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "invoices": [
                    {
                        "vendor_name": "ACME",
                        "invoice_number": "H1",
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
    history = load_vendor_history(path)
    assert len(history.history_for(_invoice("ACME", "1"))) == 1


def test_load_vendor_history_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VendorHistoryError):
        load_vendor_history(tmp_path / "nope.json")
