"""``apverify-drift`` — run the eval now and gate on regression vs a baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from apverify.eval.drift import compare, load_snapshot
from apverify.eval.report import render_drift
from apverify.eval.runner import run_eval

app = typer.Typer(add_completion=False, help="Drift detection against a saved baseline.")

_EXIT_OK = 0
_EXIT_REGRESSED = 1


@app.command()
def run(
    baseline: Annotated[Path, typer.Option(exists=True, help="Baseline snapshot JSON.")],
    count: Annotated[int, typer.Option(help="Number of synthetic invoices.")] = 25,
    catch_tolerance: Annotated[float, typer.Option(help="Allowed catch-rate drop.")] = 0.0,
    false_hold_tolerance: Annotated[float, typer.Option(help="Allowed false-hold rise.")] = 0.0,
) -> None:
    reference = load_snapshot(baseline)
    current = run_eval(count).to_snapshot()
    drift = compare(reference, current, catch_tolerance, false_hold_tolerance)

    render_drift(reference, current, drift, Console())
    raise typer.Exit(_EXIT_REGRESSED if drift.regressed else _EXIT_OK)
