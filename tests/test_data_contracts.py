import polars as pl

from credit_risk_validation.config import ColumnConfig, ValidationOptions
from credit_risk_validation.data_contracts import validate_contract
from credit_risk_validation.status import Status


def test_required_target_missing(sample_frame: pl.DataFrame) -> None:
    checks = validate_contract(
        sample_frame.drop("target"),
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert checks[0].status == Status.CRITICAL


def test_pd_out_of_range_warns_when_clipping(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns((pl.col("pd") * 3).alias("pd"))
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("pd_range") and check.status == Status.WARNING for check in checks
    )


def test_dataset_without_events_is_insufficient() -> None:
    frame = pl.DataFrame({"target": [0] * 50, "pd": [0.1] * 50})
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(check.status == Status.INSUFFICIENT_DATA for check in checks)


def test_negative_weight_is_critical(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(pl.lit(-1.0).alias("weight"))
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd", weight="weight"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("weight_positive") and check.status == Status.CRITICAL
        for check in checks
    )
