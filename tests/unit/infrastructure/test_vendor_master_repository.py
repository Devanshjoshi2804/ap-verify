from __future__ import annotations

import json
from pathlib import Path

import pytest

from apverify.application.ports import VendorMasterRepository
from apverify.infrastructure.vendor_master.repository import (
    VendorMasterError,
    load_vendor_master,
)


def test_load_vendor_master_parses_a_file(tmp_path: Path) -> None:
    path = tmp_path / "vendors.json"
    path.write_text(
        json.dumps({"vendors": [{"name": "ACME Steel Pvt Ltd", "bank_accounts": ["ACCT-0001"]}]})
    )
    master = load_vendor_master(path)
    assert isinstance(master, VendorMasterRepository)
    vendors = master.known_vendors()
    assert vendors[0].name == "ACME Steel Pvt Ltd"
    assert vendors[0].bank_accounts == frozenset({"ACCT-0001"})


def test_load_vendor_master_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VendorMasterError):
        load_vendor_master(tmp_path / "nope.json")
