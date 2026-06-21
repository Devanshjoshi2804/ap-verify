"""Load previously-posted invoices into an in-memory ledger for duplicate detection.

The duplicate detector compares a candidate against prior invoices; this adapter is the
production source of those priors, keyed by invoice number. Mirrors the other JSON
loaders and reuses the shared ``InvoiceDTO``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from apverify.domain.fraud import IdentifiedInvoice
from apverify.infrastructure.errors import AdapterError
from apverify.infrastructure.mapping import InvoiceDTO, to_domain


class _LedgerFileDTO(BaseModel):
    invoices: list[InvoiceDTO]


class InvoiceLedgerError(AdapterError):
    """Invoice-ledger data could not be loaded."""


class InMemoryInvoiceLedger:
    def __init__(self, priors: Sequence[IdentifiedInvoice]) -> None:
        self._priors = tuple(priors)

    def known_invoices(self) -> tuple[IdentifiedInvoice, ...]:
        return self._priors


def load_invoice_ledger(path: Path) -> InMemoryInvoiceLedger:
    try:
        document = _LedgerFileDTO.model_validate(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise InvoiceLedgerError(f"could not load invoice ledger from {path}: {exc}") from exc
    return InMemoryInvoiceLedger(
        [
            IdentifiedInvoice(identifier=dto.invoice_number, invoice=to_domain(dto))
            for dto in document.invoices
        ]
    )
