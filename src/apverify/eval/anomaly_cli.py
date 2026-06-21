"""``apverify-eval-anomaly`` — anomaly-detection benchmark (pure vs Isolation Forest).

Synthetic only. Isolation Forest is included when scikit-learn is installed
(``pip install -e '.[anomaly]'``); otherwise the pure detector is reported alone.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.anomaly_eval import evaluate_anomaly
from apverify.eval.anomaly_synthesis import build_anomaly_cases
from apverify.eval.report import render_anomaly
from apverify.eval.synthetic import generate_dataset

app = typer.Typer(add_completion=False, help="Anomaly-detection benchmark.")


@app.command()
def run(count: Annotated[int, typer.Option(help="Synthetic base invoices.")] = 25) -> None:
    report = evaluate_anomaly(build_anomaly_cases(generate_dataset(count=count)))
    render_anomaly(report, Console())
