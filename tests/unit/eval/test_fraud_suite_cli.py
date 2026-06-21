from __future__ import annotations

from typer.testing import CliRunner

from apverify.eval.fraud_suite_cli import app


def test_cli_runs_the_fraud_suite() -> None:
    result = CliRunner().invoke(app, ["--count", "5"])
    assert result.exit_code == 0
    assert "fraud" in result.stdout.lower()
