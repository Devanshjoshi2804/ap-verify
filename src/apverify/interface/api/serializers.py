"""Map a domain ``PayableReview`` to the HTTP response schema."""

from __future__ import annotations

from apverify.application.review_payable import PayableReview
from apverify.domain.invoice import Invoice
from apverify.interface.api.schemas import (
    AuditOut,
    CheckOut,
    ConsistencyOut,
    FieldConfidenceOut,
    InvoiceOut,
    LineItemOut,
    LineMatchOut,
    MatchFindingOut,
    MatchOut,
    ReviewResponse,
    TraceOut,
)


def to_response(review: PayableReview) -> ReviewResponse:
    report = review.critic_report
    return ReviewResponse(
        decision=str(review.decision.decision),
        reasons=list(review.decision.reasons),
        overall_confidence=report.overall_confidence,
        invoice=_invoice(review.invoice),
        fields=[
            FieldConfidenceOut(
                field=str(fc.field),
                value=fc.value,
                confidence=fc.confidence,
                checks=[
                    CheckOut(category=str(c.category), status=c.status.value, detail=c.detail)
                    for c in fc.checks
                ],
            )
            for fc in report.field_confidences
        ],
        match=MatchOut(
            outcome=str(review.match_report.outcome),
            findings=[
                MatchFindingOut(dimension=str(f.dimension), status=f.status.value, detail=f.detail)
                for f in review.match_report.findings
            ],
            line_matches=[
                LineMatchOut(
                    invoice_description=lm.invoice_description,
                    status=lm.status.value,
                    detail=lm.detail,
                )
                for lm in review.match_report.line_matches
            ],
        ),
        trace=[
            TraceOut(step=e.step, detail=e.detail, duration_ms=e.duration_ms) for e in review.trace
        ],
        audit=[
            AuditOut(
                field=str(v.field),
                trustworthy=v.trustworthy,
                confidence=v.confidence,
                reason=v.reason,
            )
            for v in review.audit_verdicts
        ],
        consistency=[
            ConsistencyOut(
                field=str(c.field),
                agreement=c.agreement.value,
                primary=c.primary,
                secondary=c.secondary,
            )
            for c in (review.consistency_report.comparisons if review.consistency_report else ())
        ],
    )


def _invoice(invoice: Invoice) -> InvoiceOut:
    return InvoiceOut(
        vendor_name=invoice.vendor_name,
        vendor_gstin=invoice.vendor_gstin,
        invoice_number=invoice.invoice_number,
        invoice_date=invoice.invoice_date,
        currency=invoice.currency,
        subtotal=str(invoice.subtotal),
        tax=str(invoice.tax),
        total=str(invoice.total),
        purchase_order_ref=invoice.purchase_order_ref,
        line_items=[
            LineItemOut(
                description=item.description,
                quantity=item.quantity,
                unit_price=str(item.unit_price),
                line_total=str(item.line_total),
                hsn_sac=item.hsn_sac,
            )
            for item in invoice.line_items
        ],
    )
