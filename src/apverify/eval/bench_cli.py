"""``apverify-bench`` — measure the verification layer's throughput at scale."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.report import render_throughput
from apverify.eval.throughput import run_throughput

app = typer.Typer(add_completion=False, help="Throughput harness for the critic.")


@app.command()
def run(
    count: Annotated[int, typer.Option(help="Number of synthetic invoices.")] = 500,
    workers: Annotated[int, typer.Option(help="Concurrent workers.")] = 1,
    corrupt_ratio: Annotated[
        float, typer.Option(help="Fraction of invoices to corrupt, for a realistic mix.")
    ] = 0.0,
) -> None:
    report = run_throughput(count=count, workers=workers, corrupt_ratio=corrupt_ratio)
    render_throughput(report, Console())
