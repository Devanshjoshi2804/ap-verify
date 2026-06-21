"""Injected-error generators — the other half of the eval.

Each corruption takes a correct invoice and returns a copy with one realistic
error, the kind a real extractor makes: a flipped digit, an invented total, a
mistyped GSTIN, a line that doesn't add up. Reviewed against the *correct* page
text, every one of these should be caught before approval; the share that is
caught is the headline hallucination-catch rate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from apverify.domain.invoice import Invoice
from apverify.domain.value_objects import Money

_GSTIN_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True, slots=True)
class Corruption:
    kind: str
    apply: Callable[[Invoice], Invoice]


def corruptions() -> tuple[Corruption, ...]:
    return (
        Corruption("flip_total_digit", _flip_total_digit),
        Corruption("hallucinate_total", _hallucinate_total),
        Corruption("swap_vendor", _swap_vendor),
        Corruption("invalid_gstin", _invalid_gstin),
        Corruption("wrong_line_total", _wrong_line_total),
        Corruption("break_subtotal", _break_subtotal),
    )


def _flip_total_digit(invoice: Invoice) -> Invoice:
    # 1,84,200 -> 1,84,000: a plausible misread that breaks subtotal + tax = total.
    return replace(invoice, total=invoice.total - Money.of("200.00"))


def _hallucinate_total(invoice: Invoice) -> Invoice:
    return replace(invoice, total=Money.of("999999.00"))


def _swap_vendor(invoice: Invoice) -> Invoice:
    return replace(invoice, vendor_name="Phantom Holdings International")


def _invalid_gstin(invoice: Invoice) -> Invoice:
    if invoice.vendor_gstin is None:
        return invoice
    return replace(invoice, vendor_gstin=_break_checksum(invoice.vendor_gstin))


def _wrong_line_total(invoice: Invoice) -> Invoice:
    if not invoice.line_items:
        return invoice
    first, *rest = invoice.line_items
    tampered = replace(first, line_total=first.line_total + Money.of("500.00"))
    return replace(invoice, line_items=(tampered, *rest))


def _break_subtotal(invoice: Invoice) -> Invoice:
    return replace(invoice, subtotal=invoice.subtotal + Money.of("300.00"))


def _break_checksum(gstin: str) -> str:
    last = gstin[-1]
    replacement = next(char for char in _GSTIN_ALPHABET if char != last)
    return gstin[:-1] + replacement
