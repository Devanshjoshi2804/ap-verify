"""``apverify-explain`` — demo of exact linear attribution on the fusion model.

Fits the interpretable fusion logistic regression on a small deterministic set of
feature rows, then shows the ranked ``weight * feature`` contributions behind one
low-trust field — the auditable "why" behind a fused score.
"""

from __future__ import annotations

import typer
from rich.console import Console

from apverify.eval.fusion import FeatureRow, explain_fusion, fit_logistic
from apverify.eval.report import render_explanation

app = typer.Typer(add_completion=False, help="Explain a fused trust score (linear attribution).")


def _rows() -> list[FeatureRow]:
    # Correctness tracks arithmetic + cross-model; the model learns to weight them.
    return [
        FeatureRow(
            label=f"d{index}",
            field="total",
            critic_confidence=0.8,
            verbalized_confidence=0.7,
            cross_check_passed=True,
            arithmetic_passed=index % 2 == 0,
            format_passed=True,
            cross_model_agrees=index % 2 == 0,
            correct=index % 2 == 0,
        )
        for index in range(12)
    ]


@app.command()
def run() -> None:
    rows = _rows()
    model = fit_logistic(rows)
    suspect = next(row for row in rows if not row.correct)
    render_explanation(explain_fusion(model, suspect), Console())
