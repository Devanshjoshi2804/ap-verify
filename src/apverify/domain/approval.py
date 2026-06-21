"""The approver: one final decision from two independent verdicts.

The critic judges whether the extraction can be *trusted*; the matcher judges
whether the invoice *agrees with what was ordered and received*. A payment is only
safe to auto-approve when both are clean, so the final decision is the more
cautious of the two — and the reasons explain exactly why anything short of
auto-approval was chosen.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.audit import AuditVerdict
from apverify.domain.consistency import ConsistencyReport
from apverify.domain.critique import (
    DEFAULT_POLICY,
    ApprovalDecision,
    CriticReport,
    InvoiceField,
    Policy,
)
from apverify.domain.fraud import DuplicateMatch, DuplicateTier
from apverify.domain.matching import MatchOutcome, MatchReport, MatchStatus
from apverify.domain.vendor_master import Severity, VendorRiskAssessment

_SEVERITY = {
    ApprovalDecision.AUTO_APPROVE: 0,
    ApprovalDecision.HUMAN_REVIEW: 1,
    ApprovalDecision.HOLD: 2,
}

_MATCH_ROUTE = {
    MatchOutcome.MATCHED: ApprovalDecision.AUTO_APPROVE,
    MatchOutcome.PARTIAL: ApprovalDecision.HUMAN_REVIEW,
    MatchOutcome.MISMATCH: ApprovalDecision.HUMAN_REVIEW,
    MatchOutcome.NO_PURCHASE_ORDER: ApprovalDecision.HUMAN_REVIEW,
}


@dataclass(frozen=True, slots=True)
class FinalDecision:
    decision: ApprovalDecision
    reasons: tuple[str, ...]


def approve(critic: CriticReport, match: MatchReport) -> FinalDecision:
    match_decision = _MATCH_ROUTE[match.outcome]
    decision = max(critic.decision, match_decision, key=lambda d: _SEVERITY[d])

    reasons: list[str] = []
    if critic.decision is not ApprovalDecision.AUTO_APPROVE:
        reasons.append(_critic_reason(critic))
    if match.outcome is not MatchOutcome.MATCHED:
        reasons.append(_match_reason(match))
    if not reasons:
        reasons.append("extraction verified and 3-way match clean")

    return FinalDecision(decision=decision, reasons=tuple(reasons))


def reconcile_with_audit(
    decision: FinalDecision,
    verdicts: Sequence[AuditVerdict],
    policy: Policy = DEFAULT_POLICY,
) -> FinalDecision:
    """Fold an LLM auditor's verdicts into an existing decision.

    Trustworthy verdicts never *raise* confidence — a second opinion can veto an
    auto-approval but not manufacture one.
    """
    distrusted = [verdict for verdict in verdicts if not verdict.trustworthy]
    return _escalate(
        decision,
        {verdict.field for verdict in distrusted},
        [f"auditor distrusts {verdict.field}: {verdict.reason}" for verdict in distrusted],
        policy,
    )


def reconcile_with_consistency(
    decision: FinalDecision,
    report: ConsistencyReport,
    policy: Policy = DEFAULT_POLICY,
) -> FinalDecision:
    """Fold a two-model self-consistency report into an existing decision.

    Where the two extractions disagree, at least one is wrong; on a critical field
    that holds the payment, elsewhere it routes to a human.
    """
    return _escalate(
        decision,
        {comparison.field for comparison in report.disagreements},
        [
            f"extractions disagree on {c.field}: {c.primary!r} vs {c.secondary!r}"
            for c in report.disagreements
        ],
        policy,
    )


def reconcile_with_vendor_risk(
    decision: FinalDecision,
    assessment: VendorRiskAssessment,
    policy: Policy = DEFAULT_POLICY,
) -> FinalDecision:
    """Fold a vendor-master / BEC assessment into an existing decision.

    A HIGH-severity flag (a changed bank account or an impersonated vendor) is the
    redirected-payment worst case, so it holds. A LOW flag (a new payee — common and
    usually legitimate) is surfaced as a reason but never blocks. Never lowers a
    decision.
    """
    reason = f"vendor-risk {assessment.kind.value}: {assessment.reason}"
    if assessment.severity is Severity.HIGH:
        held = max(decision.decision, ApprovalDecision.HOLD, key=lambda d: _SEVERITY[d])
        return FinalDecision(decision=held, reasons=(*decision.reasons, reason))
    if assessment.severity is Severity.LOW:
        return FinalDecision(decision=decision.decision, reasons=(*decision.reasons, reason))
    return decision


def reconcile_with_anomaly(
    decision: FinalDecision,
    assessment: AnomalyAssessment,
    policy: Policy = DEFAULT_POLICY,
) -> FinalDecision:
    """Fold a statistical-anomaly assessment into a decision: a HIGH anomaly holds the
    payment, a MEDIUM one routes to a human, and NONE is left untouched. Never lowers a
    decision."""
    if assessment.severity is AnomalySeverity.NONE:
        return decision
    target = (
        ApprovalDecision.HOLD
        if assessment.severity is AnomalySeverity.HIGH
        else ApprovalDecision.HUMAN_REVIEW
    )
    escalated = max(decision.decision, target, key=lambda d: _SEVERITY[d])
    reason = f"anomaly {assessment.top_feature}: {assessment.reason}"
    return FinalDecision(decision=escalated, reasons=(*decision.reasons, reason))


_DUPLICATE_TARGET = {
    DuplicateTier.EXACT_RESEND: ApprovalDecision.HOLD,
    DuplicateTier.OCR_VARIANT: ApprovalDecision.HOLD,
    DuplicateTier.NEAR_DUPLICATE: ApprovalDecision.HUMAN_REVIEW,
}


def reconcile_with_duplicate(
    decision: FinalDecision,
    match: DuplicateMatch | None,
    policy: Policy = DEFAULT_POLICY,
) -> FinalDecision:
    """Fold a duplicate-invoice match into a decision: a confirmed resend (exact or an
    OCR variant of a prior invoice) holds the payment — paying it twice is the loss; a
    looser near-duplicate routes to a human. No match leaves the decision untouched, and
    it never lowers one."""
    if match is None or match.tier not in _DUPLICATE_TARGET:
        return decision
    target = _DUPLICATE_TARGET[match.tier]
    escalated = max(decision.decision, target, key=lambda d: _SEVERITY[d])
    reason = f"duplicate {match.tier.value}: {match.reason}"
    return FinalDecision(decision=escalated, reasons=(*decision.reasons, reason))


def _escalate(
    decision: FinalDecision,
    fields: Collection[InvoiceField],
    reasons: Sequence[str],
    policy: Policy,
) -> FinalDecision:
    """Raise a decision to at least review (or hold, if a critical field is
    involved), appending the reasons. Never lowers an existing decision."""
    if not fields:
        return decision
    hits_critical = any(field in policy.critical_fields for field in fields)
    target = ApprovalDecision.HOLD if hits_critical else ApprovalDecision.HUMAN_REVIEW
    escalated = max(decision.decision, target, key=lambda d: _SEVERITY[d])
    return FinalDecision(decision=escalated, reasons=decision.reasons + tuple(reasons))


def _critic_reason(critic: CriticReport) -> str:
    # Only called when the critic withheld auto-approval, which always implies at
    # least one failed check, so there is always something to name.
    flagged = ", ".join(sorted({flag.field for flag in critic.flags}))
    return f"extraction confidence {critic.overall_confidence:.0%}; flagged: {flagged}"


def _match_reason(match: MatchReport) -> str:
    if match.outcome is MatchOutcome.NO_PURCHASE_ORDER:
        return "no purchase order to match against"
    issues = [
        f"{finding.dimension}: {finding.detail}"
        for finding in match.findings
        if finding.status is not MatchStatus.MATCHED
    ]
    issues += [
        f"line '{line.invoice_description}': {line.detail}"
        for line in match.line_matches
        if line.status is not MatchStatus.MATCHED
    ]
    return f"3-way match {match.outcome}" + (f" — {'; '.join(issues)}" if issues else "")
