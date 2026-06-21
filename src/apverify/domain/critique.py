"""The critic's vocabulary: what a check produces, how findings roll up into a
per-field confidence, and the policy that turns confidence into an approval
decision.

The decision rule is deliberately conservative. A single hallucinated critical
field (a total or GSTIN that never appears on the page) blocks auto-approval
outright, no matter how clean everything else looks — in accounts payable a
false auto-approve costs real money, while a false hold only costs a glance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, StrEnum


class InvoiceField(StrEnum):
    VENDOR = "vendor_name"
    GSTIN = "vendor_gstin"
    INVOICE_NUMBER = "invoice_number"
    INVOICE_DATE = "invoice_date"
    CURRENCY = "currency"
    SUBTOTAL = "subtotal"
    TAX = "tax"
    TOTAL = "total"
    LINE_ITEMS = "line_items"


class CheckCategory(StrEnum):
    CROSS_CHECK = "ocr_cross_check"
    ARITHMETIC = "arithmetic"
    FORMAT = "format"


class CheckStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ApprovalDecision(StrEnum):
    AUTO_APPROVE = "AUTO_APPROVE"
    HOLD = "HOLD"
    HUMAN_REVIEW = "HUMAN_REVIEW"


@dataclass(frozen=True, slots=True)
class CheckResult:
    category: CheckCategory
    field: InvoiceField
    status: CheckStatus
    detail: str

    @property
    def failed(self) -> bool:
        return self.status is CheckStatus.FAILED


@dataclass(frozen=True, slots=True)
class FieldConfidence:
    field: InvoiceField
    value: str
    confidence: float
    checks: tuple[CheckResult, ...]


@dataclass(frozen=True, slots=True)
class CriticReport:
    field_confidences: tuple[FieldConfidence, ...]
    overall_confidence: float
    decision: ApprovalDecision

    @property
    def checks(self) -> tuple[CheckResult, ...]:
        return tuple(c for fc in self.field_confidences for c in fc.checks)

    @property
    def flags(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if c.failed)


@dataclass(frozen=True, slots=True)
class Policy:
    """Tunable thresholds for the approve/hold/review decision.

    Defaults are intentionally cautious; calibrating these against a labelled set
    (the v1 eval harness) is what turns "feels safe" into a measured trust number.
    """

    auto_approve_at: float = 0.90
    human_review_at: float = 0.50
    arithmetic_tolerance: Decimal = Decimal("0.05")
    critical_fields: frozenset[InvoiceField] = field(
        default_factory=lambda: frozenset({InvoiceField.TOTAL, InvoiceField.GSTIN})
    )
    failure_penalty: dict[CheckCategory, float] = field(
        default_factory=lambda: {
            CheckCategory.CROSS_CHECK: 0.1,
            CheckCategory.ARITHMETIC: 0.3,
            CheckCategory.FORMAT: 0.5,
        }
    )


DEFAULT_POLICY = Policy()


def score_field(checks: tuple[CheckResult, ...], policy: Policy) -> float:
    """Combine a field's checks into a 0..1 confidence.

    Each failure multiplies confidence by its category penalty, so an
    independently corroborated field stays near 1.0 while a hallucinated one
    (failed cross-check) collapses toward 0.
    """
    confidence = 1.0
    for check in checks:
        if check.failed:
            confidence *= policy.failure_penalty[check.category]
    return round(confidence, 4)


def decide(field_confidences: tuple[FieldConfidence, ...], policy: Policy) -> ApprovalDecision:
    hallucinated_critical = any(
        check.failed
        and check.category is CheckCategory.CROSS_CHECK
        and fc.field in policy.critical_fields
        for fc in field_confidences
        for check in fc.checks
    )
    if hallucinated_critical:
        return ApprovalDecision.HOLD

    overall = overall_confidence(field_confidences)
    if overall >= policy.auto_approve_at:
        return ApprovalDecision.AUTO_APPROVE
    if overall >= policy.human_review_at:
        return ApprovalDecision.HUMAN_REVIEW
    return ApprovalDecision.HOLD


def overall_confidence(field_confidences: tuple[FieldConfidence, ...]) -> float:
    """An invoice is only as trustworthy as its weakest field, so take the min."""
    if not field_confidences:
        return 0.0
    return min(fc.confidence for fc in field_confidences)
