"""Preparation for Default of Credit Card Clients."""

from pathlib import Path

import polars as pl

from credit_risk_validation.datasets.baseline_model import add_baseline_pd
from credit_risk_validation.datasets.metadata import write_dataset_metadata


def prepare_default_credit_card_clients(
    raw_dir: Path, output_dir: Path, *, sample_size: int | None = None, seed: int = 42
) -> list[Path]:
    """Prepare standardized default credit card clients files."""

    candidates = list(raw_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError("Could not find raw CSV for default credit card clients")
    frame = pl.read_csv(candidates[0], infer_schema_length=None)
    rename_map = {}
    for column in frame.columns:
        if column.strip().lower() in {"default.payment.next.month", "default"}:
            rename_map[column] = "target"
    frame = frame.rename(rename_map)
    if "target" not in frame.columns:
        raise ValueError("Could not identify target column")
    if "SEX" in frame.columns:
        frame = frame.with_columns(
            pl.concat_str([pl.lit("sex_"), pl.col("SEX").cast(pl.Utf8)]).alias("sex_segment")
        )
    if sample_size:
        frame = frame.head(sample_size)
    reference, current = add_baseline_pd(frame, target_col="target", seed=seed)
    dataset_dir = output_dir / "default_credit_card_clients"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    reference_path = dataset_dir / "reference.parquet"
    current_path = dataset_dir / "current.parquet"
    reference.write_parquet(reference_path)
    current.write_parquet(current_path)
    metadata_path = write_dataset_metadata(
        dataset_dir,
        dataset_key="default_credit_card_clients",
        source="datasets/uciml/default-of-credit-card-clients-dataset",
        target_col="default payment next month",
        reference_rows=reference.height,
        current_rows=current.height,
        extra={"raw_file": str(candidates[0])},
    )
    return [reference_path, current_path, metadata_path]
