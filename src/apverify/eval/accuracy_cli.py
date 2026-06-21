"""``apverify-eval-accuracy`` — per-field + line-item extraction accuracy.

Runs the live extractor over labelled invoice images and scores precision / recall
/ F1 per field and at the line-item level against ground truth. Needs the dataset
extra, the dataset locally (DocILE) or downloadable (CORD), and a GEMINI_API_KEY
(one model call per document).
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.accuracy_eval import (
    LabelledDocument,
    load_cord_labelled,
    load_docile_labelled,
    run_field_accuracy,
)
from apverify.eval.report import render_accuracy
from apverify.interface.cli.bootstrap import build_extractor

app = typer.Typer(add_completion=False, help="Per-field + line-item extraction accuracy.")


@app.command()
def run(
    dataset: Annotated[str, typer.Option(help="cord or docile.")] = "docile",
    dataset_path: Annotated[str, typer.Option(help="DocILE path (required for docile).")] = "",
    split: Annotated[str, typer.Option(help="Dataset split.")] = "val",
    limit: Annotated[int, typer.Option(help="Documents to score (one model call each).")] = 25,
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

    console.print(f"[dim]Extracting {len(documents)} documents with the vision model…[/dim]")
    report = run_field_accuracy(documents, build_extractor())
    render_accuracy(report, f"{dataset.upper()} — per-field + line-item accuracy", console)
