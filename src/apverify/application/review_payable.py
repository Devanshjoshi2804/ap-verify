"""The full v1 pipeline: extract → verify → match → approve.

This is the orchestrator. It runs the agents in order, timing each step and
recording it to the trace (and to an injected tracer — the observability the spec
asks for), then hands the two independent verdicts to the approver for a single
decision. Pure orchestration over injected ports; all I/O lives behind them, and
the clock is injectable so the timing is deterministic under test.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from apverify.application.errors import PortError
from apverify.application.observability import NullTracer, Tracer
from apverify.application.ports import (
    AnomalyDetector,
    DocumentRenderer,
    InvoiceExtractor,
    InvoiceLedger,
    OcrTextProvider,
    PageImage,
    ProcurementRepository,
    SemanticAuditor,
    VendorHistoryRepository,
    VendorMasterRepository,
)
from apverify.domain.anomaly import AnomalyAssessment, AnomalySeverity
from apverify.domain.approval import (
    FinalDecision,
    approve,
    reconcile_with_anomaly,
    reconcile_with_audit,
    reconcile_with_consistency,
    reconcile_with_duplicate,
    reconcile_with_vendor_risk,
)
from apverify.domain.audit import AuditVerdict
from apverify.domain.checks import review
from apverify.domain.consistency import ConsistencyReport, compare_extractions
from apverify.domain.critique import DEFAULT_POLICY, CriticReport, InvoiceField, Policy
from apverify.domain.explanation import (
    Explanation,
    explain_anomaly,
    explain_duplicate,
    explain_vendor_risk,
)
from apverify.domain.fraud import DuplicateMatch, find_duplicates
from apverify.domain.invoice import Invoice
from apverify.domain.matching import (
    DEFAULT_MATCH_POLICY,
    MatchPolicy,
    MatchReport,
    three_way_match,
)
from apverify.domain.ocr import RawText
from apverify.domain.vendor_master import Severity, VendorRiskAssessment, assess_vendor_risk

_T = TypeVar("_T")


def _explanations(
    vendor_risk: VendorRiskAssessment | None,
    anomaly: AnomalyAssessment | None,
    duplicate: DuplicateMatch | None,
) -> tuple[Explanation, ...]:
    built: list[Explanation] = []
    if duplicate is not None:
        built.append(explain_duplicate(duplicate))
    if vendor_risk is not None and vendor_risk.severity is not Severity.NONE:
        built.append(explain_vendor_risk(vendor_risk))
    if anomaly is not None and anomaly.severity is not AnomalySeverity.NONE:
        built.append(explain_anomaly(anomaly))
    return tuple(built)


@dataclass(frozen=True, slots=True)
class TraceEntry:
    step: str
    detail: str
    duration_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class PayableReview:
    invoice: Invoice
    raw_text: RawText
    critic_report: CriticReport
    match_report: MatchReport
    decision: FinalDecision
    trace: tuple[TraceEntry, ...]
    audit_verdicts: tuple[AuditVerdict, ...] = ()
    consistency_report: ConsistencyReport | None = None
    vendor_risk: VendorRiskAssessment | None = None
    anomaly: AnomalyAssessment | None = None
    duplicate: DuplicateMatch | None = None
    explanations: tuple[Explanation, ...] = ()


class ReviewPayableUseCase:
    def __init__(
        self,
        renderer: DocumentRenderer,
        extractor: InvoiceExtractor,
        ocr: OcrTextProvider,
        procurement: ProcurementRepository,
        auditor: SemanticAuditor | None = None,
        secondary_extractor: InvoiceExtractor | None = None,
        vendor_master: VendorMasterRepository | None = None,
        anomaly_detector: AnomalyDetector | None = None,
        vendor_history: VendorHistoryRepository | None = None,
        invoice_ledger: InvoiceLedger | None = None,
        critic_policy: Policy = DEFAULT_POLICY,
        match_policy: MatchPolicy = DEFAULT_MATCH_POLICY,
        audit_below: float = 0.95,
        tracer: Tracer | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._renderer = renderer
        self._extractor = extractor
        self._ocr = ocr
        self._procurement = procurement
        self._auditor = auditor
        self._secondary_extractor = secondary_extractor
        self._vendor_master = vendor_master
        self._anomaly_detector = anomaly_detector
        self._vendor_history = vendor_history
        self._invoice_ledger = invoice_ledger
        self._critic_policy = critic_policy
        self._match_policy = match_policy
        self._audit_below = audit_below
        self._tracer = tracer or NullTracer()
        self._clock = clock

    def execute(self, document: Path) -> PayableReview:
        trace: list[TraceEntry] = []

        pages, elapsed = self._timed(lambda: self._renderer.render(document))
        self._record(trace, "render", f"{len(pages)} page(s)", elapsed)

        invoice, elapsed = self._timed(lambda: self._extractor.extract(pages))
        self._record(trace, "extract", f"{invoice.vendor_name} · total {invoice.total}", elapsed)

        raw_text, elapsed = self._timed(lambda: self._ocr.read(pages))
        self._record(trace, "ocr", f"{len(raw_text.text)} chars", elapsed)

        critic_report, elapsed = self._timed(lambda: review(invoice, raw_text, self._critic_policy))
        self._record(
            trace,
            "critic",
            f"{critic_report.decision} @ {critic_report.overall_confidence:.0%}",
            elapsed,
        )

        match_report, elapsed = self._timed(lambda: self._match(invoice))
        self._record(trace, "match", str(match_report.outcome), elapsed)

        decision = approve(critic_report, match_report)

        verdicts = self._audit(invoice, raw_text, critic_report, trace)
        if verdicts:
            decision = reconcile_with_audit(decision, verdicts, self._critic_policy)

        consistency = self._consistency(invoice, pages, trace)
        if consistency is not None:
            decision = reconcile_with_consistency(decision, consistency, self._critic_policy)

        vendor_risk = self._vendor_risk(invoice, trace)
        if vendor_risk is not None:
            decision = reconcile_with_vendor_risk(decision, vendor_risk, self._critic_policy)

        anomaly = self._anomaly(invoice, trace)
        if anomaly is not None:
            decision = reconcile_with_anomaly(decision, anomaly, self._critic_policy)

        duplicate = self._duplicate(invoice, trace)
        if duplicate is not None:
            decision = reconcile_with_duplicate(decision, duplicate, self._critic_policy)

        self._record(trace, "approve", str(decision.decision), 0.0)

        return PayableReview(
            invoice=invoice,
            raw_text=raw_text,
            critic_report=critic_report,
            match_report=match_report,
            decision=decision,
            trace=tuple(trace),
            audit_verdicts=verdicts,
            consistency_report=consistency,
            vendor_risk=vendor_risk,
            anomaly=anomaly,
            duplicate=duplicate,
            explanations=_explanations(vendor_risk, anomaly, duplicate),
        )

    def _match(self, invoice: Invoice) -> MatchReport:
        purchase_order = self._procurement.purchase_order_for(invoice)
        goods_receipt = (
            self._procurement.goods_receipt_for(purchase_order) if purchase_order else None
        )
        return three_way_match(invoice, purchase_order, goods_receipt, self._match_policy)

    def _audit(
        self,
        invoice: Invoice,
        raw_text: RawText,
        critic_report: CriticReport,
        trace: list[TraceEntry],
    ) -> tuple[AuditVerdict, ...]:
        """Audit only the fields the critic is unsure about — the expensive LLM
        call is reserved for where the cheap checks left genuine doubt."""
        auditor = self._auditor
        if auditor is None:
            return ()
        suspect: list[InvoiceField] = [
            confidence.field
            for confidence in critic_report.field_confidences
            if confidence.confidence < self._audit_below
        ]
        if not suspect:
            return ()
        verdicts, elapsed = self._timed(lambda: tuple(auditor.audit(invoice, raw_text, suspect)))
        distrusted = sum(1 for verdict in verdicts if not verdict.trustworthy)
        self._record(trace, "audit", f"{len(verdicts)} field(s), {distrusted} distrusted", elapsed)
        return verdicts

    def _consistency(
        self, invoice: Invoice, pages: Sequence[PageImage], trace: list[TraceEntry]
    ) -> ConsistencyReport | None:
        """Re-extract with a second model and compare; disagreement means at least
        one extraction is wrong. A flaky second opinion degrades to "no signal"
        rather than blocking the primary pipeline."""
        if self._secondary_extractor is None:
            return None
        secondary = self._secondary_extractor
        try:
            extraction, elapsed = self._timed(lambda: secondary.extract(pages))
        except PortError as exc:
            self._record(trace, "consistency", f"unavailable ({exc})", 0.0)
            return None
        report = compare_extractions(invoice, extraction)
        self._record(
            trace, "consistency", f"{len(report.disagreements)} field(s) disagree", elapsed
        )
        return report

    def _vendor_risk(
        self, invoice: Invoice, trace: list[TraceEntry]
    ) -> VendorRiskAssessment | None:
        """Check the invoice's vendor + bank against the master for BEC risk. Absent a
        master the step is skipped, leaving the pipeline unchanged."""
        if self._vendor_master is None:
            return None
        master = self._vendor_master
        assessment, elapsed = self._timed(
            lambda: assess_vendor_risk(invoice, master.known_vendors())
        )
        self._record(
            trace,
            "vendor-risk",
            f"{assessment.kind.value} ({assessment.severity.value})",
            elapsed,
        )
        return assessment

    def _anomaly(self, invoice: Invoice, trace: list[TraceEntry]) -> AnomalyAssessment | None:
        """Score the invoice against the vendor's history. Skipped unless both a
        detector and a history source are configured, leaving the pipeline unchanged."""
        if self._anomaly_detector is None or self._vendor_history is None:
            return None
        detector, history = self._anomaly_detector, self._vendor_history
        assessment, elapsed = self._timed(
            lambda: detector.score(invoice, history.history_for(invoice))
        )
        self._record(
            trace, "anomaly", f"{assessment.severity.value} ({assessment.top_feature})", elapsed
        )
        return assessment

    def _duplicate(self, invoice: Invoice, trace: list[TraceEntry]) -> DuplicateMatch | None:
        """Compare the invoice against the ledger of prior invoices. Absent a ledger the
        step is skipped; otherwise the best non-DISTINCT match (if any) is returned."""
        if self._invoice_ledger is None:
            return None
        ledger = self._invoice_ledger
        matches, elapsed = self._timed(lambda: find_duplicates(invoice, ledger.known_invoices()))
        best = matches[0] if matches else None
        detail = best.tier.value if best is not None else "none"
        self._record(trace, "duplicate", detail, elapsed)
        return best

    def _timed(self, produce: Callable[[], _T]) -> tuple[_T, float]:
        start = self._clock()
        value = produce()
        return value, (self._clock() - start) * 1000.0

    def _record(self, trace: list[TraceEntry], step: str, detail: str, duration_ms: float) -> None:
        entry = TraceEntry(step=step, detail=detail, duration_ms=round(duration_ms, 3))
        trace.append(entry)
        self._tracer.span(step, detail, entry.duration_ms)
