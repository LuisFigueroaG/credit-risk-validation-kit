"""Simple baseline model and harness for public datasets."""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk_validation.config import PDValidationConfig
from credit_risk_validation.suite import PDValidationSuite


def add_baseline_pd(
    frame: pl.DataFrame,
    *,
    target_col: str = "target",
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Entrena un baseline logistico para generar PD reproducible."""

    numeric_cols = [
        column
        for column, dtype in zip(frame.columns, frame.dtypes, strict=True)
        if column != target_col and dtype.is_numeric()
    ]
    if not numeric_cols:
        raise ValueError("No numeric features available for baseline model")
    x = frame.select(numeric_cols).fill_null(0).to_numpy()
    y = frame[target_col].to_numpy()
    train_idx, holdout_idx = train_test_split(
        np.arange(len(y)), test_size=0.4, random_state=seed, stratify=y
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x[train_idx], y[train_idx])
    pd_values = model.predict_proba(x)[:, 1]
    clipped_pd = np.clip(pd_values, 1e-6, 1 - 1e-6)
    score_values = -np.log(clipped_pd / (1 - clipped_pd))
    scored = frame.with_columns(
        pl.Series("pd", pd_values),
        pl.Series("score", score_values),
    ).with_row_index("__row_id")
    train_ids = np.asarray(train_idx).tolist()
    holdout_ids = np.asarray(holdout_idx).tolist()
    reference = scored.filter(pl.col("__row_id").is_in(train_ids)).drop("__row_id")
    current = scored.filter(pl.col("__row_id").is_in(holdout_ids)).drop("__row_id")
    return reference, current


def run_dataset_harness(dataset_key: str, *, sample_size: int | None = 5000) -> str:
    """Run validation for prepared dataset files."""

    dataset_dir = Path("data/processed") / dataset_key
    reference_path = dataset_dir / "reference.parquet"
    current_path = dataset_dir / "current.parquet"
    config_path = Path("examples/configs") / f"{dataset_key}.yml"
    if not reference_path.exists() or not current_path.exists():
        return "skipped: processed reference/current files not found"
    reference = pl.read_parquet(reference_path)
    current = pl.read_parquet(current_path)
    if sample_size:
        reference = reference.head(sample_size)
        current = current.head(sample_size)
    config = PDValidationConfig.from_yaml(config_path)
    result = PDValidationSuite.from_config(config).run(
        reference_data=reference, current_data=current
    )
    reports_dir = Path("reports") / dataset_key
    result.to_html(reports_dir / "pd_validation_report.html")
    result.to_json(reports_dir / "pd_validation_metrics.json")
    result.to_tables(reports_dir / "tables")
    return f"status={result.status.value}"
