"""``apverify-eval-bec`` — vendor-master / bank-change / impersonation benchmark.

Synthetic only: DocILE ground truth carries no bank-account data, so a BEC benchmark
cannot be built from it.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.bec_eval import evaluate_bec
from apverify.eval.bec_synthesis import build_bec_cases
from apverify.eval.report import render_bec
from apverify.eval.synthetic import generate_dataset

app = typer.Typer(add_completion=False, help="Vendor-master / BEC detection benchmark.")


@app.command()
def run(
    count: Annotated[int, typer.Option(help="Synthetic base invoices.")] = 25,
    threshold: Annotated[float, typer.Option(help="Impersonation name-match threshold.")] = 0.85,
) -> None:
    report = evaluate_bec(build_bec_cases(generate_dataset(count=count)), threshold=threshold)
    render_bec(report, Console())
