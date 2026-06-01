"""Command line interface for CRVK."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from credit_risk_validation._version import __version__
from credit_risk_validation.config import PDValidationConfig
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
    validation_config = PDValidationConfig.from_yaml(config)
    if language:
        validation_config.report.language = language
    suite = PDValidationSuite.from_config(validation_config)
    console.print(f"Loading reference: {reference}")
    reference_df = read_table(reference)
    console.print(f"Reference rows: {reference_df.height}")
    current_df = read_table(current) if current else None
    if current_df is not None:
        console.print(f"Current rows: {current_df.height}")
    result = suite.run(reference_data=reference_df, current_data=current_df)

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
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset key or Kaggle slug.")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")] = Path("data/raw"),
    unzip: Annotated[bool, typer.Option("--unzip/--no-unzip")] = True,
) -> None:
    """Download a Kaggle dataset or competition resource."""

    path = download_kaggle_resource(dataset, output_dir=output_dir, unzip=unzip)
    console.print(f"Downloaded: {path}")


@datasets_app.command("prepare")
def datasets_prepare(
    dataset: Annotated[str, typer.Option("--dataset", help="Dataset key.")],
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = Path("data/raw"),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/processed"),
    sample_size: Annotated[int | None, typer.Option("--sample-size")] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Prepare a standardized reference/current split."""

    normalized = _normalize_dataset_key(dataset)
    if normalized == "give_me_some_credit":
        paths = prepare_give_me_some_credit(raw_dir, output_dir, sample_size=sample_size, seed=seed)
    elif normalized == "default_credit_card_clients":
        paths = prepare_default_credit_card_clients(
            raw_dir, output_dir, sample_size=sample_size, seed=seed
        )
    elif normalized == "home_credit_stability":
        paths = prepare_home_credit_stability(
            raw_dir, output_dir, sample_size=sample_size, seed=seed
        )
    else:
        raise typer.BadParameter(f"Unknown dataset key: {dataset}")
    for path in paths:
        console.print(f"Prepared: {path}")


@datasets_app.command("run-harness")
def datasets_run_harness(
    dataset: Annotated[str | None, typer.Option("--dataset", help="Dataset key.")] = None,
    all_datasets: Annotated[
        bool, typer.Option("--all", help="Run all configured datasets.")
    ] = False,
    sample_size: Annotated[int | None, typer.Option("--sample-size")] = 5000,
) -> None:
    """Run dataset validation harness if processed files are available."""

    from credit_risk_validation.datasets.baseline_model import run_dataset_harness

    keys = ["give_me_some_credit", "default_credit_card_clients", "home_credit_stability"]
    selected = keys if all_datasets else [_normalize_dataset_key(dataset or "give_me_some_credit")]
    for key in selected:
        result = run_dataset_harness(key, sample_size=sample_size)
        console.print(f"{key}: {result}")


def _normalize_dataset_key(dataset: str) -> str:
    if dataset == "home_credit_model_stability":
        return "home_credit_stability"
    return dataset
