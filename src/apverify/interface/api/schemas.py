"""HTTP response schemas.

A stable JSON contract for the frontend, decoupled from the domain types.
Amounts are serialised as strings to preserve exactness across the wire.
"""

from __future__ import annotations

from pydantic import BaseModel


class CheckOut(BaseModel):
    category: str
    status: str
    detail: str


class FieldConfidenceOut(BaseModel):
    field: str
    value: str
    confidence: float
    checks: list[CheckOut]


class LineItemOut(BaseModel):
    description: str
    quantity: int
    unit_price: str
    line_total: str
    hsn_sac: str | None


class InvoiceOut(BaseModel):
    vendor_name: str
    vendor_gstin: str | None
    invoice_number: str
    invoice_date: str
    currency: str
    subtotal: str
    tax: str
    total: str
    purchase_order_ref: str | None
    line_items: list[LineItemOut]


class MatchFindingOut(BaseModel):
    dimension: str
    status: str
    detail: str


class LineMatchOut(BaseModel):
    invoice_description: str
    status: str
    detail: str


class MatchOut(BaseModel):
    outcome: str
    findings: list[MatchFindingOut]
    line_matches: list[LineMatchOut]


class TraceOut(BaseModel):
    step: str
    detail: str
    duration_ms: float


class AuditOut(BaseModel):
    field: str
    trustworthy: bool
    confidence: float
    reason: str


class ConsistencyOut(BaseModel):
    field: str
    agreement: str
    primary: str
    secondary: str


class ReviewResponse(BaseModel):
    decision: str
    reasons: list[str]
    overall_confidence: float
    invoice: InvoiceOut
    fields: list[FieldConfidenceOut]
    match: MatchOut
    trace: list[TraceOut]
    audit: list[AuditOut]
    consistency: list[ConsistencyOut]
