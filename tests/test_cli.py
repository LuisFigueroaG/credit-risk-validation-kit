from pathlib import Path

import polars as pl
import pytest
import typer
from typer.testing import CliRunner

import credit_risk_validation.cli as cli_module
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


def test_cli_pd_validate_accepts_csv(
    sample_frame: pl.DataFrame, config_path: Path, tmp_path: Path
) -> None:
    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    sample_frame.write_csv(reference)
    sample_frame.write_csv(current)
    result = CliRunner().invoke(
        app,
        [
            "pd-validate",
            "--config",
            str(config_path),
            "--reference",
            str(reference),
            "--current",
            str(current),
            "--output-json",
            str(tmp_path / "metrics.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "metrics.json").exists()


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_rejects_unsupported_language(
    sample_frame: pl.DataFrame, config_path: Path, tmp_path: Path
) -> None:
    reference = tmp_path / "reference.parquet"
    sample_frame.write_parquet(reference)
    result = CliRunner().invoke(
        app,
        [
            "pd-validate",
            "--config",
            str(config_path),
            "--reference",
            str(reference),
            "--language",
            "fr",
        ],
    )
    assert result.exit_code != 0
    assert "language must be one of" in result.output


def test_cli_datasets_download_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_download(dataset: str, *, output_dir: Path, unzip: bool) -> Path:
        calls.append(dataset)
        return output_dir

    monkeypatch.setattr(cli_module, "download_kaggle_resource", fake_download)
    result = CliRunner().invoke(
        app,
        ["datasets", "download", "--all", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert calls == cli_module.DATASET_KEYS


def test_cli_datasets_prepare_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_prepare(
        dataset: str,
        *,
        raw_dir: Path,
        output_dir: Path,
        sample_size: int | None,
        seed: int,
    ) -> list[Path]:
        calls.append(dataset)
        return [output_dir / dataset / "reference.parquet"]

    monkeypatch.setattr(cli_module, "_prepare_dataset", fake_prepare)
    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "prepare",
            "--all",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "processed"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == cli_module.DATASET_KEYS


def test_cli_datasets_run_harness_can_download_and_prepare(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str, str | None]] = []

    def fake_download(dataset: str, *, output_dir: Path, unzip: bool) -> Path:
        calls.append(("download", dataset, str(output_dir)))
        return output_dir

    def fake_prepare(
        dataset: str,
        *,
        raw_dir: Path,
        output_dir: Path,
        sample_size: int | None,
        seed: int,
    ) -> list[Path]:
        calls.append(("prepare", dataset, str(output_dir)))
        return [output_dir / dataset / "reference.parquet"]

    def fake_run_harness(dataset: str, *, sample_size: int | None, processed_dir: Path) -> str:
        calls.append(("run", dataset, str(processed_dir)))
        return "status=OK; artifacts=ok; metrics=ok"

    monkeypatch.setattr(cli_module, "download_kaggle_resource", fake_download)
    monkeypatch.setattr(cli_module, "_prepare_dataset", fake_prepare)
    monkeypatch.setattr(cli_module, "run_dataset_harness", fake_run_harness)

    result = CliRunner().invoke(
        app,
        [
            "datasets",
            "run-harness",
            "--dataset",
            "give_me_some_credit",
            "--download",
            "--prepare",
            "--raw-dir",
            str(tmp_path / "raw"),
            "--output-dir",
            str(tmp_path / "processed"),
            "--sample-size",
            "123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        ("download", "give_me_some_credit", str(tmp_path / "raw" / "give_me_some_credit")),
        ("prepare", "give_me_some_credit", str(tmp_path / "processed")),
        ("run", "give_me_some_credit", str(tmp_path / "processed")),
    ]
    assert "status=OK; artifacts=ok; metrics=ok" in result.output


def test_cli_datasets_download_requires_dataset_or_all() -> None:
    result = CliRunner().invoke(app, ["datasets", "download"])
    assert result.exit_code != 0
    assert "Usage:" in result.output


def test_selected_dataset_keys_requires_dataset_or_all() -> None:
    with pytest.raises(typer.BadParameter, match="Pass --dataset or --all"):
        cli_module._selected_dataset_keys("", False, default=None)
