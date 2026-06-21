from __future__ import annotations

from typer.testing import CliRunner

from apverify.eval.bec_cli import app


def test_cli_runs_the_bec_benchmark() -> None:
    result = CliRunner().invoke(app, ["--count", "5"])
    assert result.exit_code == 0
    assert "BEC" in result.stdout or "vendor" in result.stdout.lower()
