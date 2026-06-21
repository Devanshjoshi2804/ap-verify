"""``apverify-eval`` — run the benchmark and gate CI on it.

Exits non-zero when the catch rate or false-hold rate breach the thresholds, so a
regression in the critic fails the pipeline rather than shipping silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.drift import save_snapshot
from apverify.eval.report import render_console, render_markdown
from apverify.eval.runner import run_eval

app = typer.Typer(add_completion=False, help="Offline evaluation harness for the critic.")

_EXIT_OK = 0
_EXIT_REGRESSED = 1


@app.command()
def run(
    count: Annotated[int, typer.Option(help="Number of synthetic invoices.")] = 25,
    min_catch_rate: Annotated[float, typer.Option(help="Fail below this catch rate.")] = 0.99,
    max_false_hold: Annotated[float, typer.Option(help="Fail above this false-hold rate.")] = 0.0,
    markdown: Annotated[bool, typer.Option(help="Emit the README benchmark table.")] = False,
    save: Annotated[
        Path | None, typer.Option(help="Write a baseline snapshot for drift checks.")
    ] = None,
) -> None:
    report = run_eval(count)
    console = Console()

    if markdown:
        console.print(render_markdown(report))
    else:
        render_console(report, console)

    if save is not None:
        save_snapshot(report.to_snapshot(), save)
        console.print(f"\nSnapshot written to {save}")

    passed = report.catch_rate >= min_catch_rate and report.false_hold_rate <= max_false_hold
    if not passed:
        console.print(
            f"\n[red]Eval gate failed[/red]: catch {report.catch_rate:.1%} "
            f"(min {min_catch_rate:.0%}), false-hold {report.false_hold_rate:.1%} "
            f"(max {max_false_hold:.0%})"
        )
        raise typer.Exit(_EXIT_REGRESSED)
    raise typer.Exit(_EXIT_OK)
