"""Load the vendor master from a JSON file into an in-memory repository."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from apverify.domain.vendor_master import KnownVendor
from apverify.infrastructure.errors import AdapterError


class _KnownVendorDTO(BaseModel):
    name: str
    bank_accounts: list[str]
    gstin: str = ""


class _VendorMasterFileDTO(BaseModel):
    vendors: list[_KnownVendorDTO]


class VendorMasterError(AdapterError):
    """Vendor-master data could not be loaded."""


class InMemoryVendorMaster:
    def __init__(self, vendors: Sequence[KnownVendor]) -> None:
        self._vendors = tuple(vendors)

    def known_vendors(self) -> tuple[KnownVendor, ...]:
        return self._vendors


def load_vendor_master(path: Path) -> InMemoryVendorMaster:
    try:
        document = _VendorMasterFileDTO.model_validate(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise VendorMasterError(f"could not load vendor master from {path}: {exc}") from exc
    return InMemoryVendorMaster(
        [KnownVendor(dto.name, frozenset(dto.bank_accounts), dto.gstin) for dto in document.vendors]
    )
