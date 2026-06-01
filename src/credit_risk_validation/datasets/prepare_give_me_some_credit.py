"""Preparation for Give Me Some Credit."""

from pathlib import Path

import polars as pl

from credit_risk_validation.datasets.baseline_model import add_baseline_pd
from credit_risk_validation.datasets.metadata import write_dataset_metadata


def prepare_give_me_some_credit(
    raw_dir: Path, output_dir: Path, *, sample_size: int | None = None, seed: int = 42
) -> list[Path]:
    """Prepare standardized files from Kaggle raw CSVs."""

    candidates = list(raw_dir.rglob("cs-training.csv")) + list(raw_dir.rglob("*training*.csv"))
    if not candidates:
        raise FileNotFoundError("Could not find Give Me Some Credit training CSV")
    frame = pl.read_csv(candidates[0]).rename({"SeriousDlqin2yrs": "target"})
    frame = frame.drop([column for column in ["", "Id"] if column in frame.columns])
    if sample_size:
        frame = frame.head(sample_size)
    reference, current = add_baseline_pd(frame, target_col="target", seed=seed)
    dataset_dir = output_dir / "give_me_some_credit"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    reference_path = dataset_dir / "reference.parquet"
    current_path = dataset_dir / "current.parquet"
    reference.write_parquet(reference_path)
    current.write_parquet(current_path)
    metadata_path = write_dataset_metadata(
        dataset_dir,
        dataset_key="give_me_some_credit",
        source="competitions/GiveMeSomeCredit",
        target_col="SeriousDlqin2yrs",
        reference_rows=reference.height,
        current_rows=current.height,
        extra={"raw_file": str(candidates[0])},
    )
    return [reference_path, current_path, metadata_path]
