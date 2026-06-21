"""``apverify-eval-fraud`` — duplicate-fraud catch-rate vs false-positive benchmark.

Synthetic is the controlled headline (exact ground truth, crafted hard negatives);
DocILE is a realism check that builds invoices from ground-truth fields, so it needs
no model calls.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.domain.invoice import Invoice
from apverify.eval.accuracy_eval import load_docile_labelled
from apverify.eval.fraud_eval import evaluate_fraud, invoice_from_labelled
from apverify.eval.fraud_synthesis import FraudCase, build_fraud_cases
from apverify.eval.report import render_fraud
from apverify.eval.synthetic import GroundTruth, generate_dataset

app = typer.Typer(add_completion=False, help="Duplicate-fraud detection benchmark.")


@app.command()
def run(
    dataset: Annotated[str, typer.Option(help="synthetic or docile.")] = "synthetic",
    count: Annotated[int, typer.Option(help="Synthetic base invoices.")] = 25,
    dataset_path: Annotated[str, typer.Option(help="DocILE path (required for docile).")] = "",
    split: Annotated[str, typer.Option(help="Dataset split.")] = "val",
    limit: Annotated[int, typer.Option(help="DocILE documents.")] = 50,
    threshold: Annotated[
        float, typer.Option(help="Flag threshold (default: zero-FP operating point).")
    ] = -1.0,
) -> None:
    console = Console()
    if dataset == "docile":
        if not dataset_path:
            raise typer.BadParameter("--dataset-path is required for docile")
        documents = load_docile_labelled(dataset_path, split=split, limit=limit)
        base = [invoice_from_labelled(document) for document in documents]
        cases = _cases_from_invoices(base)
    else:
        cases = build_fraud_cases(generate_dataset(count=count))

    report = evaluate_fraud(cases, threshold=None if threshold < 0 else threshold)
    render_fraud(report, console)


def _cases_from_invoices(invoices: list[Invoice]) -> list[FraudCase]:
    """Wrap real invoices as a GroundTruth-like base so build_fraud_cases can inject."""
    base = [
        GroundTruth(label=f"docile-{index:03d}", invoice=invoice)
        for index, invoice in enumerate(invoices)
    ]
    return build_fraud_cases(base)
