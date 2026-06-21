"""Render an :class:`EvalReport` for humans and for the README."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from apverify.domain.explanation import Explanation
from apverify.eval.accuracy import AccuracyReport
from apverify.eval.anomaly_eval import AnomalyReport
from apverify.eval.bec_eval import BecReport
from apverify.eval.calibration import (
    Sample,
    best_operating_point,
    expected_calibration_error,
    reliability_bins,
    risk_coverage,
)
from apverify.eval.collusion_eval import CollusionReport
from apverify.eval.dataset_eval import DatasetReport
from apverify.eval.document_gate import DocumentGateEvaluation
from apverify.eval.drift import DriftReport
from apverify.eval.fraud_eval import FraudReport
from apverify.eval.fraud_suite_eval import FraudSuiteReport
from apverify.eval.fusion import FusionEvaluation, auroc
from apverify.eval.fusion_cv import FusionCrossValidation
from apverify.eval.metrics import EvalReport, EvalSnapshot
from apverify.eval.throughput import ThroughputReport
from apverify.eval.uncertainty_eval import UncertaintySignals


def render_accuracy(report: AccuracyReport, title: str, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title=title, title_justify="left")
    table.add_column("Field", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Support", justify="right")
    for stat in report.stats:
        table.add_row(
            stat.field,
            f"{stat.precision:.2f}",
            f"{stat.recall:.2f}",
            f"{stat.f1:.2f}",
            str(stat.support),
        )
    console.print(table)
    console.print(f"\n{report.documents} documents · macro-F1 {report.macro_f1:.2f}")
    if report.line_items is not None:
        line = report.line_items
        console.print(
            f"Line items (LIR): P {line.precision:.2f} · R {line.recall:.2f} · F1 {line.f1:.2f} "
            f"({line.matched} matched, {line.spurious} spurious, {line.missed} missed)"
        )


def render_console(report: EvalReport, console: Console | None = None) -> None:
    console = console or Console()

    headline = Table(title="Benchmark — critic on injected errors", title_justify="left")
    headline.add_column("Metric", style="cyan")
    headline.add_column("Value", justify="right")
    headline.add_row(
        "Invoices (clean / corrupted)", f"{report.clean_count} / {report.corrupt_count}"
    )
    headline.add_row("Hallucination-catch rate", f"{report.catch_rate:.1%}")
    headline.add_row("Safe-auto-approval rate", f"{report.safe_auto_approval_rate:.1%}")
    headline.add_row("False-hold rate", f"{report.false_hold_rate:.1%}")
    headline.add_row("Escaped (corrupt auto-approved)", str(report.escaped))
    console.print(headline)

    by_kind = Table(title="Catch rate by corruption", title_justify="left")
    by_kind.add_column("Corruption", style="cyan")
    by_kind.add_column("Caught", justify="right")
    for score in report.per_kind():
        by_kind.add_row(score.kind, f"{score.caught}/{score.total} ({score.catch_rate:.0%})")
    console.print(by_kind)


def render_throughput(report: ThroughputReport, console: Console | None = None) -> None:
    console = console or Console()

    table = Table(title="Throughput — critic over a synthetic batch", title_justify="left")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Invoices", str(report.invoices))
    table.add_row("Workers", str(report.workers))
    table.add_row("Wall time", f"{report.wall_seconds:.3f}s")
    table.add_row("Throughput", f"{report.throughput_per_second:,.0f}/s")
    table.add_row("Projected/day", f"{report.projected_per_day:,}")
    latencies = f"{report.p50_ms:.2f} / {report.p95_ms:.2f} / {report.max_ms:.2f} ms"
    table.add_row("Latency p50 / p95 / max", latencies)
    console.print(table)

    mix = Table(title="Decision mix", title_justify="left")
    mix.add_column("Decision", style="cyan")
    mix.add_column("Count", justify="right")
    for decision, count in sorted(report.decisions.items()):
        mix.add_row(decision, str(count))
    console.print(mix)


def render_calibration(samples: list[Sample], console: Console | None = None) -> None:
    console = console or Console()
    if not samples:
        console.print("[yellow]No calibration samples collected.[/yellow]")
        return

    ece = expected_calibration_error(samples)
    point = best_operating_point(samples)
    console.print(
        f"[bold]{len(samples)}[/bold] field predictions · "
        f"ECE [bold]{ece:.3f}[/bold] (0 = perfectly calibrated)"
    )

    reliability = Table(title="Reliability — confidence vs actual accuracy", title_justify="left")
    reliability.add_column("Confidence bin", style="cyan")
    reliability.add_column("Mean conf", justify="right")
    reliability.add_column("Accuracy", justify="right")
    reliability.add_column("Count", justify="right")
    for binned in reliability_bins(samples):
        reliability.add_row(
            f"{binned.lower:.1f}-{binned.upper:.1f}",
            f"{binned.confidence:.2f}",
            f"{binned.accuracy:.2f}",
            str(binned.count),
        )
    console.print(reliability)

    curve = Table(title="Risk-coverage", title_justify="left")
    curve.add_column("Threshold", style="cyan")
    curve.add_column("Coverage", justify="right")
    curve.add_column("Error", justify="right")
    for cp in risk_coverage(samples):
        curve.add_row(f"{cp.threshold:.1f}", f"{cp.coverage:.0%}", f"{cp.error:.0%}")
    console.print(curve)

    if point.error == 0.0 and point.coverage > 0.0:
        console.print(
            f"\nOperating point: confidence ≥ [bold]{point.threshold:.2f}[/bold] "
            f"auto-approves [bold]{point.coverage:.0%}[/bold] of fields, [green]0 wrong[/green]."
        )
    else:
        console.print(
            f"\nBest operating point: confidence ≥ [bold]{point.threshold:.2f}[/bold] → "
            f"[bold]{point.coverage:.0%}[/bold] coverage at [bold]{point.error:.0%}[/bold] error "
            "(no error-free threshold — some wrong extractions are high-confidence)."
        )


def render_uncertainty(signals: UncertaintySignals, console: Console | None = None) -> None:
    console = console or Console()
    rows = [
        ("self-consistency", signals.self_consistency),
        ("semantic-entropy", signals.semantic_entropy),
    ]
    if not any(samples for _, samples in rows):
        console.print("[yellow]No uncertainty samples collected.[/yellow]")
        return

    table = Table(
        title="Sampling-based uncertainty — does self-agreement predict correctness?",
        title_justify="left",
    )
    table.add_column("Signal", style="cyan")
    table.add_column("AUROC", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("Fields", justify="right")
    for name, samples in rows:
        table.add_row(
            name,
            f"{auroc(samples):.3f}",
            f"{expected_calibration_error(samples):.3f}",
            str(len(samples)),
        )
    console.print(table)
    console.print(
        "[dim]AUROC ~0.5 ⇒ the signal does not separate right from wrong on this set.[/dim]"
    )


def render_fusion(evaluation: FusionEvaluation, console: Console | None = None) -> None:
    console = console or Console()

    if len(evaluation.primaries) > 1:
        console.print(
            f"[bold red]⚠ rows pool multiple primary extractors "
            f"({', '.join(evaluation.primaries)}) — a fusion fit is only valid per "
            f"extractor; do not trust these numbers.[/bold red]"
        )
    elif evaluation.primaries:
        console.print(f"[dim]Primary extractor: [bold]{evaluation.primaries[0]}[/bold].[/dim]")

    diagnostic = evaluation.diagnostic
    console.print(
        f"[bold]Cross-model diagnostic[/bold]: of "
        f"[bold]{diagnostic.high_confidence_errors}[/bold] confidently-wrong fields, "
        f"a second model disagrees on [bold]{diagnostic.caught_by_disagreement}[/bold] "
        f"([green]{diagnostic.catch_rate:.0%}[/green] caught by an independent signal)."
    )
    console.print(
        f"[dim]Trained on {evaluation.train_size} fields, "
        f"evaluated on a held-out {evaluation.test_size}.[/dim]\n"
    )

    comparison = Table(
        title="Signal comparison (held-out test) — discrimination AND calibration",
        title_justify="left",
    )
    comparison.add_column("Signal", style="cyan")
    comparison.add_column("AUROC", justify="right")
    comparison.add_column("ECE", justify="right")
    comparison.add_column("Operating point", justify="left")
    for signal in evaluation.signals:
        point = signal.operating_point
        operating = (
            f"≥{point.threshold:.2f} → {point.coverage:.0%} cov @ {point.error:.0%} err"
            if point.coverage > 0.0
            else "no usable threshold"
        )
        comparison.add_row(signal.name, f"{signal.auroc:.3f}", f"{signal.ece:.3f}", operating)
    console.print(comparison)
    console.print(
        "[dim]AUROC = separates right from wrong · ECE = probabilities are honest · "
        f"fused temperature T = {evaluation.temperature:.2f} (fit on train).[/dim]"
    )

    weights = Table(title="Fusion weights (interpretable)", title_justify="left")
    weights.add_column("Signal", style="cyan")
    weights.add_column("Weight", justify="right")
    for name, weight in sorted(
        evaluation.coefficients.items(), key=lambda item: abs(item[1]), reverse=True
    ):
        weights.add_row(name, f"{weight:+.2f}")
    console.print(weights)


def render_fraud(report: FraudReport, console: Console | None = None) -> None:
    console = console or Console()
    if report.case_count == 0:
        console.print("[yellow]No fraud cases to evaluate.[/yellow]")
        return

    console.print(
        f"[bold]Duplicate-fraud detection[/bold] (n={report.case_count}, "
        f"{report.fraud_count} fraudulent): at score ≥{report.threshold:.2f}, "
        f"[green]{report.catch_rate:.0%}[/green] of duplicates caught at "
        f"[green]{report.false_positive_rate:.0%}[/green] false-positive "
        f"(precision {report.precision:.0%}, AUROC {report.auroc:.3f})."
    )

    by_kind = Table(title="Catch rate by kind", title_justify="left")
    by_kind.add_column("Kind", style="cyan")
    by_kind.add_column("Flagged", justify="right")
    for kind, rate in report.per_kind.items():
        by_kind.add_row(kind, f"{rate:.0%}")
    console.print(by_kind)

    curve = Table(title="Catch-rate vs false-positive sweep", title_justify="left")
    curve.add_column("Score ≥", justify="right")
    curve.add_column("Caught", justify="right")
    curve.add_column("False-pos", justify="right")
    for point in report.sweep:
        curve.add_row(
            f"{point.threshold:.2f}",
            f"{point.catch_rate:.0%}",
            f"{point.false_positive_rate:.0%}",
        )
    console.print(curve)


def render_anomaly(report: AnomalyReport, console: Console | None = None) -> None:
    console = console or Console()
    if report.case_count == 0:
        console.print("[yellow]No anomaly cases to evaluate.[/yellow]")
        return

    note = (
        ""
        if report.sklearn_available
        else " [dim]scikit-learn not installed — pure detector only.[/dim]"
    )
    console.print(
        f"[bold]Anomaly detection[/bold] (n={report.case_count}, "
        f"{report.anomaly_count} anomalous).{note}"
    )
    table = Table(title="Detector comparison", title_justify="left")
    table.add_column("Detector", style="cyan")
    table.add_column("AUROC", justify="right")
    table.add_column("Caught", justify="right")
    table.add_column("False-pos", justify="right")
    for result in report.results:
        table.add_row(
            result.name,
            f"{result.auroc:.3f}",
            f"{result.catch_rate:.0%}",
            f"{result.false_positive_rate:.0%}",
        )
    console.print(table)


def render_collusion(report: CollusionReport, console: Console | None = None) -> None:
    console = console or Console()
    if report.pair_count == 0:
        console.print("[yellow]No approver-vendor pairs to evaluate.[/yellow]")
        return
    console.print(
        f"[bold]Collusion detection[/bold] ({report.pair_count} pairs, "
        f"{report.colluding_count} colluding): caught "
        f"[green]{report.catch_rate:.0%}[/green] at "
        f"[green]{report.false_positive_rate:.0%}[/green] false-positive "
        f"(precision {report.precision:.0%}, AUROC {report.auroc:.3f})."
    )


def render_explanation(explanation: Explanation, console: Console | None = None) -> None:
    console = console or Console()
    console.print(f"[bold]{explanation.source}[/bold] — {explanation.headline}")
    table = Table(title="Ranked factors", title_justify="left")
    table.add_column("Signal", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Contribution", justify="right")
    table.add_column("Detail")
    for factor in explanation.factors:
        table.add_row(factor.signal, factor.value, f"{factor.contribution:+.3f}", factor.detail)
    console.print(table)


def render_fraud_suite(report: FraudSuiteReport, console: Console | None = None) -> None:
    console = console or Console()
    if report.case_count == 0:
        console.print("[yellow]No fraud-suite cases to evaluate.[/yellow]")
        return

    console.print(
        f"[bold]Cross-detector fraud suite[/bold] (n={report.case_count}, "
        f"{report.fraud_count} fraudulent): the combined layer catches "
        f"[green]{report.catch_rate:.0%}[/green] of fraud at "
        f"[green]{report.false_positive_rate:.0%}[/green] false-positive "
        f"(precision {report.precision:.0%})."
    )

    by_label = Table(title="Catch rate by fraud type", title_justify="left")
    by_label.add_column("Label", style="cyan")
    by_label.add_column("Flagged", justify="right")
    for label, rate in report.per_label.items():
        by_label.add_row(label, f"{rate:.0%}")
    console.print(by_label)

    by_detector = Table(title="Frauds caught per detector", title_justify="left")
    by_detector.add_column("Detector", style="cyan")
    by_detector.add_column("Caught", justify="right")
    for detector, count in report.per_detector.items():
        by_detector.add_row(detector, str(count))
    console.print(by_detector)


def render_bec(report: BecReport, console: Console | None = None) -> None:
    console = console or Console()
    if report.case_count == 0:
        console.print("[yellow]No BEC cases to evaluate.[/yellow]")
        return

    console.print(
        f"[bold]Vendor-master / BEC detection[/bold] (n={report.case_count}): at name "
        f"threshold {report.threshold:.2f}, [green]{report.catch_rate:.0%}[/green] of "
        f"bank-change + impersonation caught at "
        f"[green]{report.false_positive_rate:.0%}[/green] false-positive "
        f"(precision {report.precision:.0%}, impersonation AUROC "
        f"{report.impersonation_auroc:.3f})."
    )

    table = Table(title="HIGH-flag rate by scenario", title_justify="left")
    table.add_column("Scenario", style="cyan")
    table.add_column("Flagged HIGH", justify="right")
    for scenario, rate in report.per_kind.items():
        table.add_row(scenario, f"{rate:.0%}")
    console.print(table)


def render_fusion_cv(evaluation: FusionCrossValidation, console: Console | None = None) -> None:
    console = console or Console()

    if len(evaluation.primaries) > 1:
        console.print(
            f"[bold red]⚠ rows pool multiple primary extractors "
            f"({', '.join(evaluation.primaries)}) — cross-validate per extractor; "
            f"do not trust these numbers.[/bold red]"
        )

    if evaluation.folds < 2:
        console.print("[yellow]Too few rows to cross-validate.[/yellow]")
        return

    table = Table(
        title=f"{evaluation.folds}-fold cross-validation (every row scored once, held-out)",
        title_justify="left",
    )
    table.add_column("Signal", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Std", justify="right")
    for metric in (evaluation.auroc, evaluation.ece):
        # AUROC and ECE are bounded [0, 1]; the t-interval can poke past the bound on a
        # small, skewed sample, so clamp the displayed range to the metric's domain.
        low = max(0.0, metric.ci_low)
        high = min(1.0, metric.ci_high)
        table.add_row(
            metric.name,
            f"{metric.mean:.3f}",
            f"[{low:.3f}, {high:.3f}]",
            f"{metric.std:.3f}",
        )
    console.print(table)
    console.print(
        f"[dim]Across {evaluation.row_count} rows · a wide CI means the single-split "
        f"number is not yet stable (accumulate more rows).[/dim]"
    )


def render_document_gate(
    evaluation: DocumentGateEvaluation, console: Console | None = None
) -> None:
    console = console or Console()

    if len(evaluation.primaries) > 1:
        console.print(
            f"[bold red]⚠ rows pool multiple primary extractors "
            f"({', '.join(evaluation.primaries)}) — a gate is only valid per extractor; "
            f"do not trust these numbers.[/bold red]"
        )
    elif evaluation.primaries:
        console.print(f"[dim]Primary extractor: [bold]{evaluation.primaries[0]}[/bold].[/dim]")

    if evaluation.test_documents == 0:
        console.print("[yellow]No held-out documents to gate.[/yellow]")
        return

    point = evaluation.operating_point
    console.print(
        f"[bold]Per-invoice auto-approval gate[/bold]: at trust ≥{point.threshold:.2f}, "
        f"[green]{point.coverage:.0%}[/green] of invoices auto-post at "
        f"[green]{point.error:.0%}[/green] document error."
    )
    console.print(
        f"[dim]Fit on {evaluation.train_documents} documents, evaluated on a held-out "
        f"{evaluation.test_documents} ({evaluation.field_count} fields). "
        f"Document trust = weakest field; a document is correct only if every field is. "
        f"AUROC {evaluation.auroc:.3f} · ECE {evaluation.ece:.3f}.[/dim]\n"
    )

    table = Table(
        title="Document risk-coverage — auto-approve the most invoices at zero error",
        title_justify="left",
    )
    table.add_column("Trust ≥", justify="right")
    table.add_column("Auto-posted", justify="right")
    table.add_column("Doc error", justify="right")
    for coverage_point in evaluation.curve:
        if coverage_point.coverage == 0.0:
            continue
        table.add_row(
            f"{coverage_point.threshold:.2f}",
            f"{coverage_point.coverage:.0%}",
            f"{coverage_point.error:.0%}",
        )
    console.print(table)

    if evaluation.autonomy:
        autonomy = Table(
            title="Selective autonomy — most invoices auto-approvable per error budget",
            title_justify="left",
        )
        autonomy.add_column("Error budget", justify="right")
        autonomy.add_column("Auto-posted", justify="right")
        autonomy.add_column("Actual doc error", justify="right")
        autonomy.add_column("Trust ≥", justify="right")
        for budget, autonomy_point in evaluation.autonomy:
            autonomy.add_row(
                f"≤{budget:.0%}",
                f"{autonomy_point.coverage:.0%}",
                f"{autonomy_point.error:.1%}",
                f"{autonomy_point.threshold:.2f}",
            )
        console.print(autonomy)


def render_drift(
    baseline: EvalSnapshot,
    candidate: EvalSnapshot,
    drift: DriftReport,
    console: Console | None = None,
) -> None:
    console = console or Console()

    table = Table(title="Drift vs baseline", title_justify="left")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Now", justify="right")
    table.add_column("Δ", justify="right")
    table.add_row(
        "Catch rate",
        f"{baseline.catch_rate:.1%}",
        f"{candidate.catch_rate:.1%}",
        _delta(drift.catch_rate_delta),
    )
    table.add_row(
        "False-hold rate",
        f"{baseline.false_hold_rate:.1%}",
        f"{candidate.false_hold_rate:.1%}",
        _delta(-drift.false_hold_delta),
    )
    table.add_row(
        "Safe-auto rate",
        f"{baseline.safe_auto_approval_rate:.1%}",
        f"{candidate.safe_auto_approval_rate:.1%}",
        _delta(drift.safe_auto_delta),
    )
    console.print(table)

    if drift.regressed:
        console.print("\n[bold red]Regression detected[/bold red]")
        for reason in drift.reasons:
            console.print(f"  [red]✗[/red] {reason}")
    else:
        console.print("\n[bold green]No regression vs baseline[/bold green]")


def _delta(value: float) -> str:
    if value > 0:
        return f"[green]+{value:.1%}[/green]"
    if value < 0:
        return f"[red]{value:.1%}[/red]"
    return "—"


def render_dataset(report: DatasetReport, title: str, console: Console | None = None) -> None:
    console = console or Console()

    summary = Table(title=title, title_justify="left")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Receipts", str(report.total))
    summary.add_row("Auto-approve rate", f"{report.auto_approve_rate:.1%}")
    for decision, count in sorted(report.decisions.items()):
        summary.add_row(f"  {decision}", str(count))
    console.print(summary)

    if report.failed_checks:
        breakdown = Table(title="Most common flags (why receipts get held)", title_justify="left")
        breakdown.add_column("Field · check", style="cyan")
        breakdown.add_column("Count", justify="right")
        ranked = sorted(report.failed_checks.items(), key=lambda kv: kv[1], reverse=True)
        for label, count in ranked[:8]:
            breakdown.add_row(label, str(count))
        console.print(breakdown)


def render_markdown(report: EvalReport) -> str:
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Invoices (clean / corrupted) | {report.clean_count} / {report.corrupt_count} |",
        f"| Hallucination-catch rate | {report.catch_rate:.1%} |",
        f"| Safe-auto-approval rate | {report.safe_auto_approval_rate:.1%} |",
        f"| False-hold rate | {report.false_hold_rate:.1%} |",
        f"| Escaped (corrupt auto-approved) | {report.escaped} |",
    ]
    return "\n".join(lines)
