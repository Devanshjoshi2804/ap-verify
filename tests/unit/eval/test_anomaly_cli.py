from __future__ import annotations

from typer.testing import CliRunner

from apverify.eval.anomaly_cli import app


def test_cli_runs_the_anomaly_benchmark() -> None:
    result = CliRunner().invoke(app, ["--count", "5"])
    assert result.exit_code == 0
    assert "anomaly" in result.stdout.lower()
