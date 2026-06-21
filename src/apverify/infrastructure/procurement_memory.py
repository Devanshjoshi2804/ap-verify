"""In-memory procurement repository.

A stand-in for the ERP/master-data lookup a real deployment would hit, seeded from
supplied records and keyed on the normalised PO number so trivial formatting
differences ("PO-2025-1001" vs "po2025 1001") still resolve.
"""

from __future__ import annotations

from collections.abc import Iterable

from apverify.domain.invoice import Invoice
from apverify.domain.procurement import GoodsReceiptNote, PurchaseOrder


class InMemoryProcurementRepository:
    def __init__(
        self,
        purchase_orders: Iterable[PurchaseOrder] = (),
        goods_receipts: Iterable[GoodsReceiptNote] = (),
    ) -> None:
        self._purchase_orders = {_canonical(po.po_number): po for po in purchase_orders}
        self._goods_receipts = {_canonical(grn.purchase_order_ref): grn for grn in goods_receipts}

    def purchase_order_for(self, invoice: Invoice) -> PurchaseOrder | None:
        if invoice.purchase_order_ref is None:
            return None
        return self._purchase_orders.get(_canonical(invoice.purchase_order_ref))

    def goods_receipt_for(self, purchase_order: PurchaseOrder) -> GoodsReceiptNote | None:
        return self._goods_receipts.get(_canonical(purchase_order.po_number))


def _canonical(reference: str) -> str:
    return "".join(char for char in reference.lower() if char.isalnum())
