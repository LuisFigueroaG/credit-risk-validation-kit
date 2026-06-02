"""Command line interface for CRVK."""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from pydantic import ValidationError
from rich.console import Console

from credit_risk_validation._version import __version__
from credit_risk_validation.config import PDValidationConfig, ReportConfig
from credit_risk_validation.datasets.baseline_model import run_dataset_harness
from credit_risk_validation.datasets.kaggle import download_kaggle_resource
from credit_risk_validation.datasets.prepare_default_credit_card_clients import (
    prepare_default_credit_card_clients,
)
from credit_risk_validation.datasets.prepare_give_me_some_credit import prepare_give_me_some_credit
from credit_risk_validation.datasets.prepare_home_credit_stability import (
    prepare_home_credit_stability,
)
from credit_risk_validation.suite import PDValidationSuite
from credit_risk_validation.utils.dataframe import read_table
from credit_risk_validation.utils.logging import configure_logging

console = Console()
app = typer.Typer(help="Credit Risk Validation Kit CLI.")
datasets_app = typer.Typer(help="Dataset download and harness commands.")
app.add_typer(datasets_app, name="datasets")

DATASET_KEYS = ["give_me_some_credit", "default_credit_card_clients", "home_credit_stability"]


@app.command("version")
def version() -> None:
    """Print the installed CRVK version."""

    console.print(__version__)


@app.command("pd-validate")
def pd_validate(
    config: Annotated[Path, typer.Option("--config", "-c", help="YAML validation config.")],
    reference: Annotated[Path, typer.Option("--reference", "-r", help="Reference CSV/Parquet.")],
    current: Annotated[Path | None, typer.Option("--current", help="Current CSV/Parquet.")] = None,
    output_html: Annotated[
        Path | None, typer.Option("--output-html", help="Path for HTML report.")
    ] = None,
    output_json: Annotated[
        Path | None, typer.Option("--output-json", help="Path for JSON metrics.")
    ] = None,
    output_tables: Annotated[
        Path | None, typer.Option("--output-tables", help="Directory for aggregate tables.")
    ] = None,
    output_model_card: Annotated[
        Path | None, typer.Option("--output-model-card", help="Path for Markdown model card.")
    ] = None,
    language: Annotated[
        str | None, typer.Option("--language", help="Override report language from config.")
    ] = None,
    fail_on_critical: Annotated[
        bool, typer.Option("--fail-on-critical", help="Exit non-zero on CRITICAL status.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable verbose logging.")] = False,
) -> None:
    """Validate binary PD model outputs."""

    configure_logging(verbose)
    logger.info("Starting PD validation CLI run")
    validation_config = PDValidationConfig.from_yaml(config)
    if language:
        try:
            validation_config.report = ReportConfig(
                **{**validation_config.report.model_dump(), "language": language}
            )
        except ValidationError as exc:
            raise typer.BadParameter("language must be one of: en, es") from exc
    suite = PDValidationSuite.from_config(validation_config)
    logger.info(
        "Validation columns configured: target={target}; pd={pd}; score_configured={score}; "
        "period_configured={period}; weight_configured={weight}; segments={segments}",
        target=validation_config.columns.target,
        pd=validation_config.columns.pd,
        score=validation_config.columns.score is not None,
        period=validation_config.columns.period is not None,
        weight=validation_config.columns.weight is not None,
        segments=len(validation_config.columns.segments),
    )
    console.print(f"Loading reference: {reference}")
    reference_df = read_table(reference)
    logger.info(
        "Loaded reference dataset: rows={rows}; columns={columns}",
        rows=reference_df.height,
        columns=reference_df.width,
    )
    console.print(f"Reference rows: {reference_df.height}")
    current_df = read_table(current) if current else None
    if current_df is not None:
        logger.info(
            "Loaded current dataset: rows={rows}; columns={columns}",
            rows=current_df.height,
            columns=current_df.width,
        )
        console.print(f"Current rows: {current_df.height}")
    result = suite.run(reference_data=reference_df, current_data=current_df)
    warning_count = sum(1 for check in result.checks if check.status.value == "WARNING") + sum(
        1 for metric in result.metrics.values() if metric.status.value == "WARNING"
    )
    critical_count = sum(1 for check in result.checks if check.status.value == "CRITICAL") + sum(
        1 for metric in result.metrics.values() if metric.status.value == "CRITICAL"
    )
    logger.info(
        "Finished PD validation CLI run: status={status}; warnings={warnings}; criticals={criticals}",
        status=result.status.value,
        warnings=warning_count,
        criticals=critical_count,
    )

    if output_html:
        result.to_html(output_html)
        console.print(f"HTML report: {output_html}")
    if output_json:
        result.to_json(output_json)
        console.print(f"JSON metrics: {output_json}")
    if output_tables:
        result.to_tables(output_tables)
        console.print(f"Tables: {output_tables}")
    if output_model_card:
        result.to_model_card(output_model_card)
        console.print(f"Model card: {output_model_card}")

    console.print(f"Status: {result.status.value}")
    if fail_on_critical and result.status.value == "CRITICAL":
        raise typer.Exit(2)


@datasets_app.command("download")
def datasets_download(
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset key or Kaggle slug.")] = "",
    all_datasets: Annotated[
        bool, typer.Option("--all", help="Download all configured datasets.")
    ] = False,
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("data/raw"),
    unzip: Annotated[bool, typer.Option("--unzip/--no-unzip")] = True,
) -> None:
    """Download a Kaggle dataset or competition resource."""

    selected = _selected_dataset_keys(dataset, all_datasets, default=None)
    for key in selected:
        path = download_kaggle_resource(key, output_dir=output_dir / key, unzip=unzip)
        console.print(f"Downloaded: {path}")


@datasets_app.command("prepare")
def datasets_prepare(
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset key.")] = "",
    all_datasets: Annotated[
        bool, typer.Option("--all", help="Prepare all configured datasets.")
    ] = False,
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = Path("data/raw"),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/processed"),
    sample_size: Annotated[int | None, typer.Option("--sample-size")] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Prepare a standardized reference/current split."""

    for normalized in _selected_dataset_keys(dataset, all_datasets, default=None):
        paths = _prepare_dataset(
            normalized,
            raw_dir=raw_dir,
            output_dir=output_dir,
            sample_size=sample_size,
            seed=seed,
        )
        for path in paths:
            console.print(f"Prepared: {path}")


@datasets_app.command("run-harness")
def datasets_run_harness(
    dataset: Annotated[str | None, typer.Option("--dataset", help="Dataset key.")] = None,
    all_datasets: Annotated[
        bool, typer.Option("--all", help="Run all configured datasets.")
    ] = False,
    sample_size: Annotated[int | None, typer.Option("--sample-size")] = 5000,
    download: Annotated[
        bool, typer.Option("--download", help="Download selected Kaggle datasets first.")
    ] = False,
    prepare: Annotated[
        bool, typer.Option("--prepare", help="Prepare selected datasets before validation.")
    ] = False,
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = Path("data/raw"),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/processed"),
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Run the dataset validation harness, optionally downloading and preparing inputs first."""

    selected = _selected_dataset_keys(dataset, all_datasets, default="give_me_some_credit")
    for key in selected:
        if download:
            download_kaggle_resource(key, output_dir=raw_dir / key, unzip=True)
            console.print("Downloaded configured dataset")
        if prepare:
            for path in _prepare_dataset(
                key,
                raw_dir=raw_dir,
                output_dir=output_dir,
                sample_size=sample_size,
                seed=seed,
            ):
                console.print(f"Prepared: {path}")
        result = run_dataset_harness(key, sample_size=sample_size, processed_dir=output_dir)
        console.print(f"{key}: {result}")


def _selected_dataset_keys(
    dataset: str | None, all_datasets: bool, *, default: str | None
) -> list[str]:
    if all_datasets:
        return DATASET_KEYS
    if not dataset:
        if default is None:
            raise typer.BadParameter("Pass --dataset or --all")
        dataset = default
    normalized = _normalize_dataset_key(dataset)
    if normalized not in DATASET_KEYS and "/" not in normalized:
        raise typer.BadParameter(f"Unknown dataset key: {dataset}")
    return [normalized]


def _prepare_dataset(
    dataset: str,
    *,
    raw_dir: Path,
    output_dir: Path,
    sample_size: int | None,
    seed: int,
) -> list[Path]:
    if dataset == "give_me_some_credit":
        return prepare_give_me_some_credit(raw_dir, output_dir, sample_size=sample_size, seed=seed)
    if dataset == "default_credit_card_clients":
        return prepare_default_credit_card_clients(
            raw_dir, output_dir, sample_size=sample_size, seed=seed
        )
    if dataset == "home_credit_stability":
        return prepare_home_credit_stability(
            raw_dir, output_dir, sample_size=sample_size, seed=seed
        )
    raise typer.BadParameter(f"Unknown dataset key: {dataset}")


def _normalize_dataset_key(dataset: str) -> str:
    if dataset == "home_credit_model_stability":
        return "home_credit_stability"
    return dataset
