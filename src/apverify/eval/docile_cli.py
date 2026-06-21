"""``apverify-eval-docile`` — run the critic over the DocILE invoice benchmark.

Requires the dataset extra (``pip install -e '.[docile]'``) and a locally
downloaded DocILE dataset (access-gated; request at docile.rossum.ai). The mapping
is an unverified scaffold — see ``apverify.eval.docile``.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.dataset_eval import run_dataset_eval
from apverify.eval.docile import load_docile
from apverify.eval.report import render_dataset

app = typer.Typer(add_completion=False, help="Critic eval over the DocILE invoice benchmark.")


@app.command()
def run(
    dataset_path: Annotated[str, typer.Option(help="Path to the downloaded DocILE dataset.")],
    split: Annotated[str, typer.Option(help="Dataset split.")] = "val",
    limit: Annotated[int, typer.Option(help="Cap the number of documents.")] = 100,
) -> None:
    console = Console()
    console.print(f"[dim]Loading DocILE ({split}, up to {limit}) from {dataset_path}…[/dim]")
    examples = load_docile(dataset_path, split=split, limit=limit)
    render_dataset(run_dataset_eval(examples), "DocILE — critic on real invoices", console)
