from __future__ import annotations

from typer.testing import CliRunner

from apverify.eval.explanation_cli import app


def test_cli_prints_a_fusion_explanation() -> None:
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0
    assert "P(correct)" in result.stdout
