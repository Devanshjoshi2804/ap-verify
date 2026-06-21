"""``apverify-eval-calibration`` — is the critic's confidence calibrated?

Runs the extractor + critic over labelled invoices and reports ECE, a reliability
table and a risk-coverage curve with the zero-wrong-auto operating point. Needs the
dataset extra, the dataset (DocILE local / CORD downloadable) and a GEMINI_API_KEY.
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
from apverify.eval.calibration import (
    expected_calibration_error,
    fit_temperature,
    temperature_scaled,
)
from apverify.eval.calibration_eval import (
    collect_calibration_samples,
    collect_uncertainty_samples,
)
from apverify.eval.report import render_calibration
from apverify.interface.cli.bootstrap import build_extractor

app = typer.Typer(add_completion=False, help="Confidence calibration of the critic.")


@app.command()
def run(
    dataset: Annotated[str, typer.Option(help="cord or docile.")] = "docile",
    dataset_path: Annotated[str, typer.Option(help="DocILE path (required for docile).")] = "",
    split: Annotated[str, typer.Option(help="Dataset split.")] = "val",
    limit: Annotated[int, typer.Option(help="Documents (one model call each).")] = 40,
    calibrate: Annotated[
        bool, typer.Option(help="Fit temperature scaling and recalibrate.")
    ] = False,
    compare: Annotated[
        bool,
        typer.Option(help="Also collect the model's verbalized confidence and compare ECE."),
    ] = False,
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

    console.print(f"[dim]Extracting + scoring {len(documents)} documents…[/dim]")

    if compare:
        _render_comparison(documents, console)
        return

    samples = collect_calibration_samples(documents, build_extractor())

    if calibrate and samples:
        temperature = fit_temperature(samples)
        before = expected_calibration_error(samples)
        after = expected_calibration_error(temperature_scaled(samples, temperature))
        console.print(
            f"\n[bold]Temperature scaling[/bold]: T = {temperature:.2f} · "
            f"ECE {before:.3f} → [green]{after:.3f}[/green]\n"
        )
        render_calibration(temperature_scaled(samples, temperature), console)
    else:
        render_calibration(samples, console)


def _render_comparison(documents: list[LabelledDocument], console: Console) -> None:
    samples = collect_uncertainty_samples(documents, build_extractor())
    critic_ece = expected_calibration_error(samples.critic)
    model_ece = expected_calibration_error(samples.verbalized)
    better = "critic" if critic_ece <= model_ece else "model (verbalized)"
    console.print(
        f"\n[bold]Critic[/bold] (structural) · ECE [green]{critic_ece:.3f}[/green] "
        f"over {len(samples.critic)} fields"
    )
    console.print(
        f"[bold]Model[/bold] (verbalized) · ECE [green]{model_ece:.3f}[/green] "
        f"over {len(samples.verbalized)} fields"
    )
    console.print(f"\nBetter-calibrated signal: [bold]{better}[/bold]\n")
    console.print("[dim]Critic reliability:[/dim]")
    render_calibration(samples.critic, console)
    console.print("\n[dim]Verbalized reliability:[/dim]")
    render_calibration(samples.verbalized, console)
