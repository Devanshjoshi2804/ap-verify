"""``apverify-eval-fraud-suite`` — the combined cross-detector fraud benchmark.

Runs the duplicate, BEC, and anomaly detectors together over one synthesized stream and
reports the combined catch-rate vs false-positive. Synthetic only.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.fraud_suite_eval import evaluate_fraud_suite
from apverify.eval.fraud_suite_synthesis import build_fraud_suite
from apverify.eval.report import render_fraud_suite
from apverify.eval.synthetic import generate_dataset

app = typer.Typer(add_completion=False, help="Combined cross-detector fraud benchmark.")


@app.command()
def run(count: Annotated[int, typer.Option(help="Synthetic base invoices.")] = 25) -> None:
    report = evaluate_fraud_suite(build_fraud_suite(generate_dataset(count=count)))
    render_fraud_suite(report, Console())
