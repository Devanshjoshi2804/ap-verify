"""``apverify-eval-uncertainty`` — do sampling-based signals predict correctness?

Resamples the extractor a few times per document and reports the AUROC/ECE of
self-consistency and semantic entropy against whether the consensus value is right.
Costs ``samples`` model calls per document; needs the dataset + a GEMINI_API_KEY.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.accuracy_eval import (
    LabelledDocument,
    load_cord_labelled,
    load_docile_labelled,
)
from apverify.eval.report import render_uncertainty
from apverify.eval.uncertainty_eval import collect_uncertainty_signals
from apverify.interface.cli.bootstrap import build_extractor

app = typer.Typer(add_completion=False, help="Sampling-based uncertainty of the extractor.")


@app.command()
def run(
    dataset: Annotated[str, typer.Option(help="cord or docile.")] = "docile",
    dataset_path: Annotated[str, typer.Option(help="DocILE path (required for docile).")] = "",
    split: Annotated[str, typer.Option(help="Dataset split.")] = "val",
    limit: Annotated[int, typer.Option(help="Documents (samples model calls each).")] = 20,
    samples: Annotated[int, typer.Option(help="Resamples per document.")] = 5,
) -> None:
    console = Console()
    console.print(f"[dim]Loading {dataset} ({split}, up to {limit})…[/dim]")
    documents: list[LabelledDocument]
    if dataset == "cord":
        documents = load_cord_labelled(split=("test" if split == "val" else split), limit=limit)
    else:
        if not dataset_path:
            raise typer.BadParameter("--dataset-path is required for docile")
        documents = load_docile_labelled(dataset_path, split=split, limit=limit)

    console.print(f"[dim]Resampling {len(documents)} documents, {samples} draws each…[/dim]")
    render_uncertainty(collect_uncertainty_signals(documents, build_extractor(), samples), console)
