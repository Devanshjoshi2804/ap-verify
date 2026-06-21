"""``apverify-eval-collusion`` — behavioral collusion benchmark over an approval log."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.collusion_eval import evaluate_collusion
from apverify.eval.collusion_synthesis import build_collusion_log
from apverify.eval.report import render_collusion

app = typer.Typer(add_completion=False, help="Behavioral collusion-detection benchmark.")


@app.command()
def run(
    pairs: Annotated[int, typer.Option(help="Approver-vendor pairs.")] = 6,
    per_pair: Annotated[int, typer.Option(help="Approvals per pair.")] = 8,
) -> None:
    records, truth = build_collusion_log(pairs=pairs, per_pair=per_pair)
    render_collusion(evaluate_collusion(records, truth), Console())
