"""``apverify-eval-fusion`` — combine independent trust signals into one calibrated
score, and report both axes a single number hides.

Collecting rows needs two models (a primary with verbalized confidence and a second
extractor for cross-model agreement) plus a dataset. Because each row costs two model
calls and free-tier quota is finite, ``--save`` persists collected rows and ``--load``
re-evaluates them offline — so fusion can be analysed, and accumulated across runs,
without re-extracting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from apverify.application.ports import InvoiceExtractor, SelfReportingExtractor
from apverify.eval.accuracy_eval import (
    LabelledDocument,
    load_cord_labelled,
    load_docile_labelled,
)
from apverify.eval.document_gate import evaluate_document_gate
from apverify.eval.fusion import (
    FeatureRow,
    evaluate_fusion,
    merge_rows,
    row_from_dict,
    row_to_dict,
)
from apverify.eval.fusion_cv import cross_validate_fusion
from apverify.eval.fusion_eval import collect_feature_rows
from apverify.eval.report import render_document_gate, render_fusion, render_fusion_cv
from apverify.interface.cli.bootstrap import build_named_extractors

app = typer.Typer(add_completion=False, help="Multi-signal fusion of the critic's trust signals.")


@app.command()
def run(
    dataset: Annotated[str, typer.Option(help="cord or docile.")] = "docile",
    dataset_path: Annotated[str, typer.Option(help="DocILE path (required for docile).")] = "",
    split: Annotated[str, typer.Option(help="Dataset split.")] = "val",
    limit: Annotated[int, typer.Option(help="Documents (two model calls each).")] = 60,
    save: Annotated[str, typer.Option(help="Write collected rows to this JSON file.")] = "",
    load: Annotated[str, typer.Option(help="Evaluate rows from JSON instead of extracting.")] = "",
    append: Annotated[
        bool, typer.Option(help="Merge freshly collected rows into the --save file.")
    ] = False,
    primary: Annotated[
        str, typer.Option(help="Primary provider: gemini | mistral | groq (default: first live).")
    ] = "",
    secondary: Annotated[
        str, typer.Option(help="Cross-model provider (default: first distinct from primary).")
    ] = "",
) -> None:
    console = Console()
    rows = (
        _load_rows(load)
        if load
        else _collect_rows(dataset, dataset_path, split, limit, primary, secondary, console)
    )

    if append and save and Path(save).exists():
        rows = merge_rows(_load_rows(save), rows)
        console.print(f"[dim]Merged with existing rows → {len(rows)} total.[/dim]")

    if not rows:
        console.print("[yellow]No feature rows collected.[/yellow]")
        raise typer.Exit(code=1)
    if save:
        _save_rows(rows, save)
        console.print(f"[dim]Saved {len(rows)} rows to {save}.[/dim]")

    console.print(f"[dim]{len(rows)} feature rows.[/dim]\n")
    render_fusion(evaluate_fusion(rows), console)
    console.print()
    render_fusion_cv(cross_validate_fusion(rows), console)
    console.print()
    render_document_gate(evaluate_document_gate(rows), console)


def _collect_rows(
    dataset: str,
    dataset_path: str,
    split: str,
    limit: int,
    primary_name: str,
    secondary_name: str,
    console: Console,
) -> list[FeatureRow]:
    console.print(f"[dim]Loading {dataset} ({split}, up to {limit})…[/dim]")
    documents: list[LabelledDocument]
    if dataset == "cord":
        documents = load_cord_labelled(split=("test" if split == "val" else split), limit=limit)
    else:
        if not dataset_path:
            raise typer.BadParameter("--dataset-path is required for docile")
        documents = load_docile_labelled(dataset_path, split=split, limit=limit)

    primary, secondary, names = _resolve_providers(primary_name, secondary_name)
    console.print(f"[dim]Cross-model pair: {names[0]} (primary) vs {names[1]}.[/dim]")
    console.print(f"[dim]Extracting (two distinct models) over {len(documents)} documents…[/dim]")
    return collect_feature_rows(
        documents, primary, secondary, primary_name=names[0], secondary_name=names[1]
    )


def _resolve_providers(
    primary_name: str, secondary_name: str
) -> tuple[SelfReportingExtractor, InvoiceExtractor, tuple[str, str]]:
    available = build_named_extractors()
    if len(available) < 2:
        raise typer.BadParameter(
            "fusion needs two distinct providers — configure a second of "
            "GEMINI / MISTRAL / GROQ keys"
        )
    names = list(available)
    primary_key = primary_name or names[0]
    secondary_key = secondary_name or next(name for name in names if name != primary_key)
    if primary_key not in available or secondary_key not in available:
        raise typer.BadParameter(f"unknown provider; available: {', '.join(names)}")
    if primary_key == secondary_key:
        raise typer.BadParameter("primary and secondary must be different providers")
    primary = available[primary_key]
    if not isinstance(primary, SelfReportingExtractor):
        raise typer.BadParameter(f"{primary_key} cannot report verbalized confidence")
    return primary, available[secondary_key], (primary_key, secondary_key)


def _load_rows(path: str) -> list[FeatureRow]:
    payload = json.loads(Path(path).read_text())
    return [row_from_dict(item) for item in payload]


def _save_rows(rows: list[FeatureRow], path: str) -> None:
    Path(path).write_text(json.dumps([row_to_dict(row) for row in rows], indent=2))
