"""Load receivables from a JSON file into the in-memory repository."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ValidationError

from apverify.domain.collections import Receivable
from apverify.domain.value_objects import Money, PhoneNumber
from apverify.infrastructure.errors import AdapterError
from apverify.infrastructure.receivables_memory import InMemoryReceivablesRepository


class _ReceivableDTO(BaseModel):
    customer_name: str
    phone: str
    invoice_number: str
    amount_due: str
    currency: str = "INR"
    due_date: date


class _ReceivablesFileDTO(BaseModel):
    receivables: list[_ReceivableDTO]


class ReceivablesError(AdapterError):
    """Receivables data could not be loaded."""


def load_receivables(path: Path) -> InMemoryReceivablesRepository:
    try:
        document = _ReceivablesFileDTO.model_validate(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ReceivablesError(f"could not load receivables from {path}: {exc}") from exc

    try:
        receivables = [_to_domain(dto) for dto in document.receivables]
    except ValueError as exc:
        raise ReceivablesError(f"invalid receivable in {path}: {exc}") from exc
    return InMemoryReceivablesRepository(receivables)


def _to_domain(dto: _ReceivableDTO) -> Receivable:
    return Receivable(
        customer_name=dto.customer_name,
        phone=PhoneNumber(dto.phone),
        invoice_number=dto.invoice_number,
        amount_due=Money.of(dto.amount_due),
        currency=dto.currency,
        due_date=dto.due_date,
    )
