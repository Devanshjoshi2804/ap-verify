from __future__ import annotations

import pytest

from apverify.infrastructure.settings import Settings
from apverify.interface.cli.bootstrap import (
    _build_invoice_ledger,
    _build_vendor_history,
    _build_vendor_master,
)


def test_invoice_ledger_is_none_when_path_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("INVOICE_LEDGER_PATH", "")
    assert _build_invoice_ledger(Settings()) is None


def test_vendor_master_is_none_when_path_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("VENDOR_MASTER_PATH", "")
    assert _build_vendor_master(Settings()) is None


def test_vendor_history_is_none_when_path_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("ANOMALY_HISTORY_PATH", "")
    assert _build_vendor_history(Settings()) is None
