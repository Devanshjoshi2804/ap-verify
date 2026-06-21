from __future__ import annotations

import pytest

from tests.support import InvoiceFactory, RawTextFactory, build_invoice, build_raw_text


@pytest.fixture
def make_invoice() -> InvoiceFactory:
    return build_invoice


@pytest.fixture
def make_raw_text() -> RawTextFactory:
    return build_raw_text
