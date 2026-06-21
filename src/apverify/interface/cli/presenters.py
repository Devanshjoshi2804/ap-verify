"""Terminal rendering of a reviewed invoice.

Presentation only — it reads an ``InvoiceReview`` and writes to a console. The
shape of the output (what counts as a flag, the decision) is decided by the
domain; this module just makes it legible.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from apverify.application.process_invoice import InvoiceReview
from apverify.application.review_payable import PayableReview
from apverify.application.run_collections import CollectionsRun
from apverify.domain.collections import ReminderTier
from apverify.domain.critique import ApprovalDecision, CheckStatus, FieldConfidence
from apverify.domain.invoice import Invoice

_TIER_STYLE = {
    ReminderTier.GENTLE: "green",
    ReminderTier.FIRM: "yellow",
    ReminderTier.FINAL: "red",
}

_DECISION_STYLE = {
    ApprovalDecision.AUTO_APPROVE: "bold green",
    ApprovalDecision.HUMAN_REVIEW: "bold yellow",
    ApprovalDecision.HOLD: "bold red",
}

_STATUS_MARK = {
    CheckStatus.PASSED: "[green]pass[/green]",
    CheckStatus.FAILED: "[red]FAIL[/red]",
    CheckStatus.SKIPPED: "[dim]skip[/dim]",
}


def render_review(review: InvoiceReview, console: Console | None = None) -> None:
    console = console or Console()
    console.print(_extraction_table(review))
    console.print(_confidence_table(review.report.field_confidences))

    if review.report.flags:
        console.print("\n[bold]Critic flags[/bold]")
        for flag in review.report.flags:
            console.print(f"  [red]✗[/red] {flag.field} · {flag.category}: {flag.detail}")

    decision = review.report.decision
    style = _DECISION_STYLE[decision]
    console.print(
        f"\nOverall confidence [bold]{review.report.overall_confidence:.0%}[/bold] "
        f"→ [{style}]{decision}[/{style}]"
    )


def render_payable_review(review: PayableReview, console: Console | None = None) -> None:
    console = console or Console()
    console.print(_invoice_table(review.invoice))

    trace = Table(title="Pipeline trace", title_justify="left")
    trace.add_column("Step", style="cyan")
    trace.add_column("Outcome")
    for entry in review.trace:
        trace.add_row(entry.step, entry.detail)
    console.print(trace)

    console.print(f"\n3-way match: [bold]{review.match_report.outcome}[/bold]")

    decision = review.decision.decision
    style = _DECISION_STYLE[decision]
    console.print(f"Decision → [{style}]{decision}[/{style}]")
    for reason in review.decision.reasons:
        console.print(f"  • {reason}")


def render_collections(run: CollectionsRun, console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="Collections run", title_justify="left")
    table.add_column("Customer", style="cyan")
    table.add_column("Invoice")
    table.add_column("Tier")
    table.add_column("Status")
    for outcome in run.outcomes:
        style = _TIER_STYLE[outcome.tier]
        status = "[green]sent[/green]" if outcome.sent else f"[red]failed[/red] {outcome.detail}"
        table.add_row(
            outcome.receivable.customer_name,
            outcome.receivable.invoice_number,
            f"[{style}]{outcome.tier}[/{style}]",
            status,
        )
    console.print(table)
    console.print(f"\n{run.sent} sent · {run.failed} failed")


def _extraction_table(review: InvoiceReview) -> Table:
    return _invoice_table(review.invoice)


def _invoice_table(invoice: Invoice) -> Table:
    table = Table(title="Extracted invoice", show_header=False, title_justify="left")
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("Vendor", invoice.vendor_name)
    table.add_row("GSTIN", invoice.vendor_gstin or "—")
    table.add_row("Invoice no.", invoice.invoice_number)
    table.add_row("Date", invoice.invoice_date)
    table.add_row("Subtotal", f"{invoice.currency} {invoice.subtotal}")
    table.add_row("Tax", f"{invoice.currency} {invoice.tax}")
    table.add_row("Total", f"{invoice.currency} {invoice.total}")
    table.add_row("Line items", str(len(invoice.line_items)))
    return table


def _confidence_table(field_confidences: tuple[FieldConfidence, ...]) -> Table:
    table = Table(title="Per-field confidence", title_justify="left")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_column("Confidence", justify="right")
    table.add_column("Checks")
    for fc in field_confidences:
        marks = " ".join(_STATUS_MARK[check.status] for check in fc.checks)
        table.add_row(str(fc.field), fc.value, f"{fc.confidence:.0%}", marks)
    return table
