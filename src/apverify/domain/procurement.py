"""Procurement records the invoice is matched against.

Purchase orders and goods-receipt notes are trusted internal master data — unlike
the invoice, they are not the output of an extractor — so they are modelled with
validated value objects and assumed correct. The matcher's job is to decide
whether the *invoice* agrees with them.
"""

from __future__ import annotations

from dataclasses import dataclass

from apverify.domain.value_objects import Money


@dataclass(frozen=True, slots=True)
class PurchaseOrderLine:
    description: str
    quantity: int
    unit_price: Money


@dataclass(frozen=True, slots=True)
class PurchaseOrder:
    po_number: str
    vendor_name: str
    currency: str
    subtotal: Money
    lines: tuple[PurchaseOrderLine, ...] = ()
    vendor_gstin: str | None = None


@dataclass(frozen=True, slots=True)
class GoodsReceiptLine:
    description: str
    quantity_received: int


@dataclass(frozen=True, slots=True)
class GoodsReceiptNote:
    grn_number: str
    purchase_order_ref: str
    lines: tuple[GoodsReceiptLine, ...] = ()
