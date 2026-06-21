"""Immutable, self-validating value objects.

A value object cannot exist in an invalid state: construction either yields a
correct instance or raises. This pushes validation to the system's edge and lets
the rest of the domain treat these types as trustworthy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from apverify.domain.errors import (
    InvalidGstinError,
    InvalidInvoiceNumberError,
    InvalidMoneyError,
    InvalidPhoneNumberError,
)

_CENTS = Decimal("0.01")


@dataclass(frozen=True, order=True, slots=True)
class Money:
    """A monetary amount held as a 2-decimal ``Decimal`` — never a float.

    Floats cannot represent currency exactly; using them in an approvals system
    is how a ₹1,84,200 invoice silently becomes ₹1,84,199.99.
    """

    amount: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.amount, Decimal):
            raise InvalidMoneyError(f"amount must be Decimal, got {type(self.amount).__name__}")
        object.__setattr__(self, "amount", self.amount.quantize(_CENTS, rounding=ROUND_HALF_UP))

    @classmethod
    def of(cls, value: Decimal | int | str) -> Money:
        """Build from a Decimal, int, or numeric string (the safe float-free path)."""
        try:
            return cls(Decimal(str(value)))
        except (InvalidOperation, ValueError) as exc:
            raise InvalidMoneyError(f"cannot parse money from {value!r}") from exc

    def __add__(self, other: Money) -> Money:
        return Money(self.amount + other.amount)

    def __sub__(self, other: Money) -> Money:
        return Money(self.amount - other.amount)

    def __mul__(self, factor: Decimal | int) -> Money:
        return Money(self.amount * Decimal(factor))

    def __str__(self) -> str:
        return f"{self.amount:.2f}"


_GSTIN_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_GSTIN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True, slots=True)
class Gstin:
    """A 15-character Indian GST identification number.

    Validated by both shape (PEP-style regex) and the official Luhn mod-36
    checksum on the 15th character, so a transposed or misread digit is rejected
    rather than accepted as a plausible-looking id.
    """

    value: str

    def __post_init__(self) -> None:
        candidate = self.value.strip().upper()
        if not _GSTIN_PATTERN.match(candidate):
            raise InvalidGstinError(f"{self.value!r} does not match the GSTIN format")
        if candidate[14] != self.compute_check_digit(candidate[:14]):
            raise InvalidGstinError(f"{self.value!r} fails the GSTIN checksum")
        object.__setattr__(self, "value", candidate)

    @staticmethod
    def compute_check_digit(first_fourteen: str) -> str:
        """Luhn mod 36 over the first 14 characters, per the GSTIN spec."""
        base = len(_GSTIN_ALPHABET)
        total = 0
        factor = 2
        for char in reversed(first_fourteen):
            addend = factor * _GSTIN_ALPHABET.index(char)
            factor = 1 if factor == 2 else 2
            total += addend // base + addend % base
        return _GSTIN_ALPHABET[(base - total % base) % base]

    @property
    def state_code(self) -> str:
        return self.value[:2]

    @property
    def pan(self) -> str:
        return self.value[2:12]

    def __str__(self) -> str:
        return self.value


_E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")


@dataclass(frozen=True, slots=True)
class PhoneNumber:
    """An E.164 phone number (``+`` country code, 8 to 15 digits)."""

    value: str

    def __post_init__(self) -> None:
        candidate = self.value.strip().replace(" ", "")
        if not _E164_PATTERN.match(candidate):
            raise InvalidPhoneNumberError(f"{self.value!r} is not a valid E.164 phone number")
        object.__setattr__(self, "value", candidate)

    def __str__(self) -> str:
        return self.value


_INVOICE_NUMBER_PATTERN = re.compile(r"^[A-Za-z0-9/\-]{1,16}$")


@dataclass(frozen=True, slots=True)
class InvoiceNumber:
    """Supplier invoice number — GST Rule 46 caps this at 16 characters."""

    value: str

    def __post_init__(self) -> None:
        candidate = self.value.strip()
        if not _INVOICE_NUMBER_PATTERN.match(candidate):
            raise InvalidInvoiceNumberError(
                f"{self.value!r} is not a valid invoice number (≤16 chars, alphanumeric/-//)"
            )
        object.__setattr__(self, "value", candidate)

    def __str__(self) -> str:
        return self.value
