"""Load procurement master data (POs and GRNs) from a JSON file.

A thin adapter that maps a JSON document to domain records and into an
in-memory repository. Amounts are strings on the wire so they reach ``Money``
without passing through a float.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from apverify.domain.procurement import (
    GoodsReceiptLine,
    GoodsReceiptNote,
    PurchaseOrder,
    PurchaseOrderLine,
)
from apverify.domain.value_objects import Money
from apverify.infrastructure.errors import ProcurementError
from apverify.infrastructure.procurement_memory import InMemoryProcurementRepository


class _PurchaseOrderLineDTO(BaseModel):
    description: str
    quantity: int
    unit_price: str


class _PurchaseOrderDTO(BaseModel):
    po_number: str
    vendor_name: str
    currency: str = "INR"
    subtotal: str
    lines: list[_PurchaseOrderLineDTO] = Field(default_factory=list)
    vendor_gstin: str | None = None


class _GoodsReceiptLineDTO(BaseModel):
    description: str
    quantity_received: int


class _GoodsReceiptDTO(BaseModel):
    grn_number: str
    purchase_order_ref: str
    lines: list[_GoodsReceiptLineDTO] = Field(default_factory=list)


class _ProcurementFileDTO(BaseModel):
    purchase_orders: list[_PurchaseOrderDTO] = Field(default_factory=list)
    goods_receipts: list[_GoodsReceiptDTO] = Field(default_factory=list)


def load_procurement(path: Path) -> InMemoryProcurementRepository:
    try:
        document = _ProcurementFileDTO.model_validate(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ProcurementError(f"could not load procurement data from {path}: {exc}") from exc

    return InMemoryProcurementRepository(
        purchase_orders=[_purchase_order(dto) for dto in document.purchase_orders],
        goods_receipts=[_goods_receipt(dto) for dto in document.goods_receipts],
    )


def _purchase_order(dto: _PurchaseOrderDTO) -> PurchaseOrder:
    return PurchaseOrder(
        po_number=dto.po_number,
        vendor_name=dto.vendor_name,
        currency=dto.currency,
        subtotal=Money.of(dto.subtotal),
        lines=tuple(
            PurchaseOrderLine(line.description, line.quantity, Money.of(line.unit_price))
            for line in dto.lines
        ),
        vendor_gstin=dto.vendor_gstin,
    )


def _goods_receipt(dto: _GoodsReceiptDTO) -> GoodsReceiptNote:
    return GoodsReceiptNote(
        grn_number=dto.grn_number,
        purchase_order_ref=dto.purchase_order_ref,
        lines=tuple(
            GoodsReceiptLine(line.description, line.quantity_received) for line in dto.lines
        ),
    )
