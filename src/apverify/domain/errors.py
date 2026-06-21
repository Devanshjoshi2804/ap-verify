"""Domain error hierarchy.

Failures in the core are modelled as explicit, named exceptions so callers can
distinguish a malformed value object from an upstream extraction failure without
inspecting strings.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for every error raised by the domain layer."""


class InvalidGstinError(DomainError):
    """Raised when a string cannot be a valid GSTIN (format or checksum)."""


class InvalidMoneyError(DomainError):
    """Raised when a monetary amount cannot be represented."""


class InvalidInvoiceNumberError(DomainError):
    """Raised when an invoice number violates the GST length/charset rules."""


class InvalidPhoneNumberError(DomainError):
    """Raised when a string is not a valid E.164 phone number."""
