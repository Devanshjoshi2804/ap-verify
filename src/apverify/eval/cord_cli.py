"""``apverify-eval-cord`` — run the critic over real CORD-v2 receipts.

Requires the dataset extra (``pip install -e '.[datasets]'``) and downloads the
dataset on first run.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.cord import load_cord
from apverify.eval.dataset_eval import run_dataset_eval
from apverify.eval.report import render_dataset

app = typer.Typer(add_completion=False, help="Critic eval over the public CORD-v2 receipts.")


@app.command()
def run(
    split: Annotated[str, typer.Option(help="Dataset split: train / validation / test.")] = "test",
    limit: Annotated[int, typer.Option(help="Cap the number of receipts.")] = 100,
) -> None:
    console = Console()
    console.print(f"[dim]Loading CORD-v2 ({split}, up to {limit})…[/dim]")
    examples = load_cord(split=split, limit=limit)
    render_dataset(run_dataset_eval(examples), "CORD-v2 — critic on real receipts", console)
