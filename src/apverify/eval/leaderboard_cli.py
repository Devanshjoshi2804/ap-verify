"""``apverify-eval-leaderboard`` — cross-provider extraction accuracy.

Runs every configured extractor (Gemini / Groq / Mistral / Ollama) over the same
labelled invoices and ranks them by per-field macro-F1 and line-item F1 — the
provider trade-off on real data. Needs the dataset extra, the dataset locally
(DocILE) or downloadable (CORD), and whichever provider keys you want compared
(Ollama needs no key). One model call per document per provider.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.application.errors import PortError
from apverify.eval.accuracy import AccuracyReport
from apverify.eval.accuracy_eval import (
    LabelledDocument,
    load_cord_labelled,
    load_docile_labelled,
    run_field_accuracy,
)
from apverify.eval.leaderboard import rank_providers
from apverify.eval.report import render_leaderboard
from apverify.interface.cli.bootstrap import build_named_extractors

app = typer.Typer(add_completion=False, help="Cross-provider extraction leaderboard.")


@app.command()
def run(
    dataset: Annotated[str, typer.Option(help="cord or docile.")] = "docile",
    dataset_path: Annotated[str, typer.Option(help="DocILE path (required for docile).")] = "",
    split: Annotated[str, typer.Option(help="Dataset split.")] = "val",
    limit: Annotated[int, typer.Option(help="Documents per provider (one model call each).")] = 25,
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

    extractors = build_named_extractors()
    if not extractors:
        raise typer.BadParameter("no extractors configured — set a provider key or OLLAMA_ENABLED")

    reports: dict[str, AccuracyReport] = {}
    for provider, extractor in extractors.items():
        console.print(f"[dim]Scoring {provider} over {len(documents)} documents…[/dim]")
        try:
            reports[provider] = run_field_accuracy(documents, extractor)
        except PortError as exc:  # a provider down/exhausted shouldn't sink the whole board
            console.print(f"[yellow]Skipped {provider}: {exc}[/yellow]")

    if not reports:
        console.print("[yellow]No provider produced a score.[/yellow]")
        return
    render_leaderboard(
        rank_providers(reports), f"{dataset.upper()} — cross-provider leaderboard", console
    )
