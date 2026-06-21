from __future__ import annotations

from typer.testing import CliRunner

from apverify.eval.collusion_cli import app


def test_cli_runs_the_collusion_benchmark() -> None:
    result = CliRunner().invoke(app, ["--pairs", "6"])
    assert result.exit_code == 0
    assert "collusion" in result.stdout.lower()
