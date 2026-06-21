"""Three-way match: invoice ↔ purchase order ↔ goods-receipt note.

The invoice is trusted only as far as it agrees with what was ordered (PO) and
what arrived (GRN). Real procurement is messy — vendor names are spelled
differently, line descriptions paraphrase, deliveries arrive in instalments — so
matching is fuzzy and tolerant by design, while still refusing to bill for more
than was ordered or received.

Pure functions; no I/O. The fuzziness uses only the standard library's
``difflib`` so the matcher stays dependency-free and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from enum import StrEnum

from apverify.domain.invoice import Invoice, LineItem
from apverify.domain.procurement import GoodsReceiptNote, PurchaseOrder, PurchaseOrderLine
from apverify.domain.value_objects import Money


class MatchDimension(StrEnum):
    VENDOR = "vendor"
    PO_REFERENCE = "po_reference"
    LINE_ITEMS = "line_items"
    SUBTOTAL = "subtotal"


class MatchStatus(StrEnum):
    MATCHED = "matched"
    PARTIAL = "partial"
    MISMATCH = "mismatch"
    MISSING = "missing"


class MatchOutcome(StrEnum):
    MATCHED = "MATCHED"
    PARTIAL = "PARTIAL"
    MISMATCH = "MISMATCH"
    NO_PURCHASE_ORDER = "NO_PURCHASE_ORDER"


@dataclass(frozen=True, slots=True)
class MatchFinding:
    dimension: MatchDimension
    status: MatchStatus
    detail: str


@dataclass(frozen=True, slots=True)
class LineMatch:
    invoice_description: str
    status: MatchStatus
    detail: str


@dataclass(frozen=True, slots=True)
class MatchReport:
    outcome: MatchOutcome
    findings: tuple[MatchFinding, ...]
    line_matches: tuple[LineMatch, ...]

    @property
    def mismatches(self) -> tuple[MatchFinding, ...]:
        return tuple(f for f in self.findings if f.status is MatchStatus.MISMATCH)


@dataclass(frozen=True, slots=True)
class MatchPolicy:
    vendor_name_threshold: float = 0.85
    description_threshold: float = 0.60
    price_tolerance: float = 0.01
    subtotal_tolerance: float = 0.01


DEFAULT_MATCH_POLICY = MatchPolicy()


def three_way_match(
    invoice: Invoice,
    purchase_order: PurchaseOrder | None,
    goods_receipt: GoodsReceiptNote | None = None,
    policy: MatchPolicy = DEFAULT_MATCH_POLICY,
) -> MatchReport:
    """Match an invoice against its PO and (optionally) the goods received."""
    if purchase_order is None:
        return MatchReport(
            outcome=MatchOutcome.NO_PURCHASE_ORDER,
            findings=(
                MatchFinding(
                    MatchDimension.PO_REFERENCE,
                    MatchStatus.MISSING,
                    "no purchase order supplied to match against",
                ),
            ),
            line_matches=(),
        )

    findings = [
        _match_vendor(invoice, purchase_order, policy),
        _match_po_reference(invoice, purchase_order),
    ]
    line_matches = tuple(
        _match_line(item, purchase_order, goods_receipt, policy) for item in invoice.line_items
    )
    findings.append(_match_subtotal(invoice, purchase_order, policy))

    return MatchReport(
        outcome=_outcome(findings, line_matches),
        findings=tuple(findings),
        line_matches=line_matches,
    )


def _match_vendor(
    invoice: Invoice, purchase_order: PurchaseOrder, policy: MatchPolicy
) -> MatchFinding:
    if (
        invoice.vendor_gstin
        and purchase_order.vendor_gstin
        and invoice.vendor_gstin.strip().upper() != purchase_order.vendor_gstin.strip().upper()
    ):
        return MatchFinding(
            MatchDimension.VENDOR,
            MatchStatus.MISMATCH,
            f"GSTIN {invoice.vendor_gstin} ≠ PO GSTIN {purchase_order.vendor_gstin}",
        )

    similarity = _similarity(invoice.vendor_name, purchase_order.vendor_name)
    if similarity >= policy.vendor_name_threshold:
        return MatchFinding(
            MatchDimension.VENDOR,
            MatchStatus.MATCHED,
            f"vendor matches (similarity {similarity:.2f})",
        )
    return MatchFinding(
        MatchDimension.VENDOR,
        MatchStatus.MISMATCH,
        f"vendor {invoice.vendor_name!r} ≠ PO vendor {purchase_order.vendor_name!r}",
    )


def _match_po_reference(invoice: Invoice, purchase_order: PurchaseOrder) -> MatchFinding:
    if invoice.purchase_order_ref is None:
        return MatchFinding(
            MatchDimension.PO_REFERENCE,
            MatchStatus.MISSING,
            "invoice does not cite a purchase order",
        )
    if _canonical(invoice.purchase_order_ref) == _canonical(purchase_order.po_number):
        return MatchFinding(
            MatchDimension.PO_REFERENCE, MatchStatus.MATCHED, f"cites PO {purchase_order.po_number}"
        )
    return MatchFinding(
        MatchDimension.PO_REFERENCE,
        MatchStatus.MISMATCH,
        f"invoice cites {invoice.purchase_order_ref!r}, expected {purchase_order.po_number!r}",
    )


def _match_line(
    item: LineItem,
    purchase_order: PurchaseOrder,
    goods_receipt: GoodsReceiptNote | None,
    policy: MatchPolicy,
) -> LineMatch:
    po_line = _best_line(item.description, purchase_order.lines, policy.description_threshold)
    if po_line is None:
        return LineMatch(
            item.description, MatchStatus.MISSING, "no matching line on the purchase order"
        )

    if not _within_ratio(item.unit_price, po_line.unit_price, policy.price_tolerance):
        return LineMatch(
            item.description,
            MatchStatus.MISMATCH,
            f"unit price {item.unit_price} ≠ PO price {po_line.unit_price}",
        )

    if item.quantity > po_line.quantity:
        return LineMatch(
            item.description,
            MatchStatus.MISMATCH,
            f"billed qty {item.quantity} exceeds ordered {po_line.quantity}",
        )

    received = _received_quantity(item.description, goods_receipt, policy.description_threshold)
    if received is not None and item.quantity > received:
        return LineMatch(
            item.description,
            MatchStatus.MISMATCH,
            f"billed qty {item.quantity} exceeds received {received}",
        )
    if received is not None and received < po_line.quantity:
        return LineMatch(
            item.description,
            MatchStatus.PARTIAL,
            f"partial delivery: {received} of {po_line.quantity} received",
        )

    return LineMatch(item.description, MatchStatus.MATCHED, "qty and price within tolerance")


def _match_subtotal(
    invoice: Invoice, purchase_order: PurchaseOrder, policy: MatchPolicy
) -> MatchFinding:
    expected = _expected_subtotal(invoice, purchase_order, policy)
    if _within_ratio(invoice.subtotal, expected, policy.subtotal_tolerance):
        return MatchFinding(
            MatchDimension.SUBTOTAL,
            MatchStatus.MATCHED,
            f"subtotal {invoice.subtotal} matches PO-priced {expected}",
        )
    return MatchFinding(
        MatchDimension.SUBTOTAL,
        MatchStatus.MISMATCH,
        f"subtotal {invoice.subtotal} ≠ PO-priced {expected}",
    )


def _expected_subtotal(
    invoice: Invoice, purchase_order: PurchaseOrder, policy: MatchPolicy
) -> Money:
    """What the billed quantities should cost at PO prices."""
    total = Money.of(0)
    for item in invoice.line_items:
        po_line = _best_line(item.description, purchase_order.lines, policy.description_threshold)
        unit_price = po_line.unit_price if po_line is not None else item.unit_price
        total += unit_price * item.quantity
    return total


def _outcome(findings: list[MatchFinding], line_matches: tuple[LineMatch, ...]) -> MatchOutcome:
    line_statuses = [lm.status for lm in line_matches]
    statuses = [f.status for f in findings] + line_statuses
    if MatchStatus.MISMATCH in statuses or MatchStatus.MISSING in line_statuses:
        return MatchOutcome.MISMATCH
    if MatchStatus.PARTIAL in statuses:
        return MatchOutcome.PARTIAL
    return MatchOutcome.MATCHED


def _best_line(
    description: str, lines: tuple[PurchaseOrderLine, ...], threshold: float
) -> PurchaseOrderLine | None:
    best: PurchaseOrderLine | None = None
    best_score = threshold
    for line in lines:
        score = _similarity(description, line.description)
        if score >= best_score:
            best, best_score = line, score
    return best


def _received_quantity(
    description: str, goods_receipt: GoodsReceiptNote | None, threshold: float
) -> int | None:
    if goods_receipt is None:
        return None
    received = [
        line.quantity_received
        for line in goods_receipt.lines
        if _similarity(description, line.description) >= threshold
    ]
    return sum(received) if received else None


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.strip().lower(), right.strip().lower()).ratio()


def _canonical(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _within_ratio(actual: Money, expected: Money, tolerance: float) -> bool:
    if expected.amount == 0:
        return actual.amount == 0
    return abs(actual.amount - expected.amount) / abs(expected.amount) <= Decimal(str(tolerance))
