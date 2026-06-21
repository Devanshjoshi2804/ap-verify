from __future__ import annotations

from pathlib import Path

import pytest
from tests.support import build_invoice

from apverify.infrastructure.errors import ProcurementError
from apverify.infrastructure.procurement_loader import load_procurement

_SAMPLE = Path(__file__).resolve().parents[3] / "samples" / "procurement.json"


def test_loads_sample_procurement_and_resolves_a_purchase_order() -> None:
    repository = load_procurement(_SAMPLE)
    invoice = build_invoice(purchase_order_ref="PO-2025-1001")

    purchase_order = repository.purchase_order_for(invoice)

    assert purchase_order is not None
    assert purchase_order.vendor_name == "ACME Steel Pvt Ltd"
    assert repository.goods_receipt_for(purchase_order) is not None


def test_missing_file_raises_procurement_error() -> None:
    with pytest.raises(ProcurementError):
        load_procurement(Path("does-not-exist.json"))
