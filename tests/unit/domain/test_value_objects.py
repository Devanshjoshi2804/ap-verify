from __future__ import annotations

from decimal import Decimal

import pytest

from apverify.domain.errors import (
    InvalidGstinError,
    InvalidInvoiceNumberError,
    InvalidMoneyError,
)
from apverify.domain.value_objects import Gstin, InvoiceNumber, Money

_VALID_GSTIN_BASE = "27AABCU9603R1Z"
_VALID_GSTIN = _VALID_GSTIN_BASE + Gstin.compute_check_digit(_VALID_GSTIN_BASE)


class TestMoney:
    def test_quantises_to_two_places(self) -> None:
        assert Money.of("1.005").amount == Decimal("1.01")

    def test_parses_int_and_string(self) -> None:
        assert Money.of(100).amount == Decimal("100.00")
        assert Money.of("250.5").amount == Decimal("250.50")

    def test_adds_and_subtracts(self) -> None:
        assert Money.of("100.00") + Money.of("0.50") == Money.of("100.50")
        assert Money.of("100.00") - Money.of("0.50") == Money.of("99.50")

    def test_orders_by_amount(self) -> None:
        assert Money.of("1.00") < Money.of("2.00")

    def test_rejects_raw_float(self) -> None:
        with pytest.raises(InvalidMoneyError):
            Money(0.1)  # type: ignore[arg-type]

    def test_rejects_unparseable_value(self) -> None:
        with pytest.raises(InvalidMoneyError):
            Money.of("not-a-number")

    def test_str_keeps_two_places(self) -> None:
        assert str(Money.of("184200")) == "184200.00"


class TestGstin:
    def test_accepts_a_valid_gstin(self) -> None:
        assert Gstin(_VALID_GSTIN).value == _VALID_GSTIN

    def test_normalises_case_and_whitespace(self) -> None:
        assert Gstin(f"  {_VALID_GSTIN.lower()} ").value == _VALID_GSTIN

    def test_rejects_bad_checksum(self) -> None:
        wrong_last = "A" if _VALID_GSTIN[-1] != "A" else "B"
        with pytest.raises(InvalidGstinError, match="checksum"):
            Gstin(_VALID_GSTIN[:-1] + wrong_last)

    def test_rejects_bad_format(self) -> None:
        with pytest.raises(InvalidGstinError, match="format"):
            Gstin("GST123")

    def test_exposes_state_code_and_pan(self) -> None:
        gstin = Gstin(_VALID_GSTIN)
        assert gstin.state_code == "27"
        assert gstin.pan == "AABCU9603R"


class TestInvoiceNumber:
    def test_accepts_typical_numbers(self) -> None:
        assert InvoiceNumber("INV-2025-0042").value == "INV-2025-0042"
        assert InvoiceNumber("BT/2025/318").value == "BT/2025/318"

    def test_rejects_empty(self) -> None:
        with pytest.raises(InvalidInvoiceNumberError):
            InvoiceNumber("   ")

    def test_rejects_over_sixteen_chars(self) -> None:
        with pytest.raises(InvalidInvoiceNumberError):
            InvoiceNumber("X" * 17)
