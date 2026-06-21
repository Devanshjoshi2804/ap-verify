"""Command-line entrypoint.

Thin by design: parse one argument, build the use case at the composition root,
run it, present the result, and map the approval decision to an exit code so the
command is usable as a gate (0 = safe to auto-approve, 1 = needs attention).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console

from apverify.application.run_collections import RunCollectionsUseCase
from apverify.domain.critique import ApprovalDecision
from apverify.infrastructure.errors import AdapterError
from apverify.infrastructure.procurement_loader import load_procurement
from apverify.infrastructure.receivables_loader import load_receivables
from apverify.interface.cli.bootstrap import (
    build_message_sender,
    build_review_use_case,
    build_use_case,
)
from apverify.interface.cli.presenters import (
    render_collections,
    render_payable_review,
    render_review,
)

app = typer.Typer(
    help="Auditable accounts-payable extraction guarded by a verification critic.",
    no_args_is_help=True,
    add_completion=False,
)

_EXIT_AUTO_APPROVE = 0
_EXIT_NEEDS_ATTENTION = 1
_EXIT_ERROR = 2


@app.callback()
def _cli() -> None:
    """Keep ``run`` an explicit subcommand rather than the implicit default."""


@app.command()
def run(
    document: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Invoice PDF or image."),
    ],
) -> None:
    """Extract, verify, and decide on a single invoice."""
    console = Console()

    try:
        use_case = build_use_case()
    except ValidationError:
        console.print(
            "[red]GEMINI_API_KEY is not set.[/red] Copy .env.example to .env and add your key."
        )
        raise typer.Exit(_EXIT_ERROR) from None

    try:
        review = use_case.execute(document)
    except AdapterError as exc:
        console.print(f"[red]Processing failed:[/red] {exc}")
        raise typer.Exit(_EXIT_ERROR) from exc

    render_review(review, console)
    decision = review.report.decision
    raise typer.Exit(
        _EXIT_AUTO_APPROVE if decision is ApprovalDecision.AUTO_APPROVE else _EXIT_NEEDS_ATTENTION
    )


@app.command()
def review(
    document: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Invoice PDF or image."),
    ],
    purchase_orders: Annotated[
        Path | None,
        typer.Option(
            "--purchase-orders",
            "-p",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON file of purchase orders and goods-receipt notes to match against.",
        ),
    ] = None,
    audit: Annotated[
        bool,
        typer.Option("--audit", help="Add the Groq LLM auditor on low-confidence fields."),
    ] = False,
    cross_check: Annotated[
        bool,
        typer.Option("--cross-check", help="Re-extract with Mistral and flag disagreements."),
    ] = False,
) -> None:
    """Run the full pipeline: extract → verify → 3-way match → approve."""
    console = Console()

    try:
        repository = load_procurement(purchase_orders) if purchase_orders else None
        use_case = build_review_use_case(
            repository, enable_audit=audit, enable_cross_check=cross_check
        )
    except ValidationError:
        console.print(
            "[red]GEMINI_API_KEY is not set.[/red] Copy .env.example to .env and add your key."
        )
        raise typer.Exit(_EXIT_ERROR) from None
    except AdapterError as exc:
        console.print(f"[red]Could not start:[/red] {exc}")
        raise typer.Exit(_EXIT_ERROR) from exc

    try:
        result = use_case.execute(document)
    except AdapterError as exc:
        console.print(f"[red]Processing failed:[/red] {exc}")
        raise typer.Exit(_EXIT_ERROR) from exc

    render_payable_review(result, console)
    raise typer.Exit(
        _EXIT_AUTO_APPROVE
        if result.decision.decision is ApprovalDecision.AUTO_APPROVE
        else _EXIT_NEEDS_ATTENTION
    )


@app.command()
def collect(
    receivables: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Receivables JSON file."),
    ],
    dry_run: Annotated[
        bool, typer.Option(help="Print messages instead of sending them over WhatsApp.")
    ] = True,
    as_of: Annotated[
        str | None, typer.Option(help="Treat this ISO date as today (for demos).")
    ] = None,
) -> None:
    """Send tiered WhatsApp payment reminders for overdue receivables."""
    console = Console()
    try:
        repository = load_receivables(receivables)
        sender = build_message_sender(dry_run=dry_run)
    except AdapterError as exc:
        console.print(f"[red]Could not start collections:[/red] {exc}")
        raise typer.Exit(_EXIT_ERROR) from exc

    today = date.fromisoformat(as_of) if as_of else date.today()
    use_case = RunCollectionsUseCase(repository, sender, today=lambda: today)
    render_collections(use_case.execute(), console)
    raise typer.Exit(_EXIT_AUTO_APPROVE)
