from __future__ import annotations

from typer.testing import CliRunner

from apverify.eval.fraud_cli import app


def test_cli_runs_the_synthetic_benchmark() -> None:
    result = CliRunner().invoke(app, ["--dataset", "synthetic", "--count", "6"])
    assert result.exit_code == 0
    assert "Duplicate-fraud detection" in result.stdout
