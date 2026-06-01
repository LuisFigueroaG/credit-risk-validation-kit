from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from credit_risk_validation.cli import app


def test_cli_pd_validate(sample_frame: pl.DataFrame, config_path: Path, tmp_path: Path) -> None:
    reference = tmp_path / "reference.parquet"
    current = tmp_path / "current.parquet"
    sample_frame.write_parquet(reference)
    sample_frame.write_parquet(current)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "pd-validate",
            "--config",
            str(config_path),
            "--reference",
            str(reference),
            "--current",
            str(current),
            "--output-html",
            str(tmp_path / "report.html"),
            "--output-json",
            str(tmp_path / "metrics.json"),
            "--output-tables",
            str(tmp_path / "tables"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "metrics.json").exists()


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
