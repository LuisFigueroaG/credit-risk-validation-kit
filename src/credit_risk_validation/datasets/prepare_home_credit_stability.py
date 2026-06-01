"""Preparation for Home Credit Stability sample mode."""

from pathlib import Path

import polars as pl

from credit_risk_validation.datasets.baseline_model import add_baseline_pd
from credit_risk_validation.datasets.metadata import write_dataset_metadata


def prepare_home_credit_stability(
    raw_dir: Path, output_dir: Path, *, sample_size: int | None = 10000, seed: int = 42
) -> list[Path]:
    """Prepare a sample from Home Credit Stability if raw files are available."""

    candidates = (
        list(raw_dir.rglob("train_base.parquet"))
        + list(raw_dir.rglob("*train_base*.csv"))
        + list(raw_dir.rglob("*train*.parquet"))
        + list(raw_dir.rglob("*train*.csv"))
    )
    if not candidates:
        raise FileNotFoundError("Could not find Home Credit Stability train file")
    source = candidates[0]
    frame = pl.read_parquet(source) if source.suffix == ".parquet" else pl.read_csv(source)
    static_candidates = list(raw_dir.rglob("train_static_0_0.parquet")) + list(
        raw_dir.rglob("*train_static*.csv")
    )
    if static_candidates and "case_id" in frame.columns:
        static_source = static_candidates[0]
        static = (
            pl.read_parquet(static_source)
            if static_source.suffix == ".parquet"
            else pl.read_csv(static_source)
        )
        if "case_id" in static.columns:
            frame = frame.join(static, on="case_id", how="left")
    if "target" not in frame.columns:
        lower_map = {column: column.lower() for column in frame.columns}
        frame = frame.rename(lower_map)
    if "target" not in frame.columns:
        raise ValueError("Could not identify target column")
    if sample_size:
        frame = frame.head(sample_size)
    reference, current = add_baseline_pd(frame, target_col="target", seed=seed)
    dataset_dir = output_dir / "home_credit_stability"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    reference_path = dataset_dir / "reference.parquet"
    current_path = dataset_dir / "current.parquet"
    reference.write_parquet(reference_path)
    current.write_parquet(current_path)
    metadata_path = write_dataset_metadata(
        dataset_dir,
        dataset_key="home_credit_stability",
        source="competitions/home-credit-credit-risk-model-stability",
        target_col="target",
        reference_rows=reference.height,
        current_rows=current.height,
        extra={"raw_file": str(source)},
    )
    return [reference_path, current_path, metadata_path]
