"""Load a vendor's prior invoices — the baseline anomaly detection measures against."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from apverify.domain.invoice import Invoice
from apverify.infrastructure.errors import AdapterError
from apverify.infrastructure.mapping import InvoiceDTO, to_domain


class _HistoryFileDTO(BaseModel):
    invoices: list[InvoiceDTO]


class VendorHistoryError(AdapterError):
    """Vendor-history data could not be loaded."""


class InMemoryVendorHistory:
    def __init__(self, invoices: Sequence[Invoice]) -> None:
        self._by_vendor: dict[str, list[Invoice]] = {}
        for invoice in invoices:
            self._by_vendor.setdefault(invoice.vendor_name, []).append(invoice)

    def history_for(self, invoice: Invoice) -> tuple[Invoice, ...]:
        return tuple(self._by_vendor.get(invoice.vendor_name, ()))


def load_vendor_history(path: Path) -> InMemoryVendorHistory:
    try:
        document = _HistoryFileDTO.model_validate(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise VendorHistoryError(f"could not load vendor history from {path}: {exc}") from exc
    return InMemoryVendorHistory([to_domain(dto) for dto in document.invoices])
