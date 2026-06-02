import polars as pl
import pytest

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


def test_required_pd_missing(sample_frame: pl.DataFrame) -> None:
    checks = validate_contract(
        sample_frame.drop("pd"),
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert checks[0].status == Status.CRITICAL


def test_target_must_be_binary(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0).then(2).otherwise(pl.col("target")).alias("target")
    )
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("target_binary") and check.status == Status.CRITICAL for check in checks
    )


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


def test_pd_out_of_range_is_critical_without_clipping(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns((pl.col("pd") * 3).alias("pd"))
    options = ValidationOptions(min_events=5, min_non_events=5, min_rows=20)
    options.clip_pd.enabled = False
    checks = validate_contract(frame, ColumnConfig(target="target", pd="pd"), options)
    assert any(
        check.name.endswith("pd_range") and check.status == Status.CRITICAL for check in checks
    )


def test_dataset_without_events_is_insufficient() -> None:
    frame = pl.DataFrame({"target": [0] * 50, "pd": [0.1] * 50})
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(check.status == Status.INSUFFICIENT_DATA for check in checks)


def test_dataset_without_non_events_is_insufficient() -> None:
    frame = pl.DataFrame({"target": [1] * 50, "pd": [0.8] * 50})
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


def test_id_period_duplicates_are_critical(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(
        pl.lit("same-id").alias("customer_id"),
        pl.lit("2025-01-01").alias("period"),
    )
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd", id="customer_id", period="period"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("id_period_duplicates") and check.status == Status.CRITICAL
        for check in checks
    )


def test_period_must_be_parseable(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(pl.lit("not-a-date").alias("period"))
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd", period="period"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("period_parseable") and check.status == Status.WARNING
        for check in checks
    )


def test_polars_rejects_duplicate_columns_before_contract() -> None:
    with pytest.raises(pl.exceptions.DuplicateError):
        pl.DataFrame([[0, 0.1]], schema=["target", "target"], orient="row")


def test_target_nan_is_critical(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(float("nan"))
        .otherwise(pl.col("target").cast(pl.Float64))
        .alias("target")
    )
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("target_nan") and check.status == Status.CRITICAL for check in checks
    )


def test_pd_nan_is_critical(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0).then(float("nan")).otherwise(pl.col("pd")).alias("pd")
    )
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("pd_nan") and check.status == Status.CRITICAL for check in checks
    )


def test_weight_nan_is_critical(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0).then(float("nan")).otherwise(1.0).alias("weight")
    )
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd", weight="weight"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("weight_nan") and check.status == Status.CRITICAL for check in checks
    )


def test_score_nan_warns(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(float("nan"))
        .otherwise(pl.col("score"))
        .alias("score")
    )
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd", score="score"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("score_nan") and check.status == Status.WARNING for check in checks
    )


def test_empty_dataset_is_critical() -> None:
    frame = pl.DataFrame({"target": [], "pd": []}, schema={"target": pl.Int64, "pd": pl.Float64})
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(min_events=1, min_non_events=1, min_rows=1),
    )
    assert any(check.name.endswith("empty") and check.status == Status.CRITICAL for check in checks)


def test_constant_pd_warns(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(pl.lit(0.2).alias("pd"))
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("pd_constant") and check.status == Status.WARNING for check in checks
    )


def test_constant_score_warns(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(pl.lit(700.0).alias("score"))
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd", score="score"),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("score_constant") and check.status == Status.WARNING for check in checks
    )


def test_highly_imbalanced_target_warns() -> None:
    frame = pl.DataFrame({"target": [1, *([0] * 199)], "pd": [0.4, *([0.1] * 199)]})
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(min_events=1, min_non_events=1, min_rows=20),
    )
    assert any(
        check.name.endswith("target_imbalance") and check.status == Status.WARNING
        for check in checks
    )


def test_many_pd_values_at_boundary_warns() -> None:
    frame = pl.DataFrame(
        {
            "target": [0, 1] * 25,
            "pd": [0.0, 1.0, *([0.2, 0.8] * 24)],
        }
    )
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd"),
        ValidationOptions(
            min_events=5,
            min_non_events=5,
            min_rows=20,
            pd_boundary_warning_share=0.02,
        ),
    )
    assert any(
        check.name.endswith("pd_boundary_mass") and check.status == Status.WARNING
        for check in checks
    )


def test_high_missing_share_warns(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(
        pl.when(pl.arange(0, pl.len()) < 80)
        .then(None)
        .otherwise(pl.col("segment"))
        .alias("segment")
    )
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd", segments=["segment"]),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20, max_missing_share=0.20),
    )
    assert any(
        check.name.endswith("missing_share.segment") and check.status == Status.WARNING
        for check in checks
    )


def test_high_segment_cardinality_warns(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(
        pl.int_range(pl.len(), dtype=pl.Int64).cast(pl.Utf8).alias("segment")
    )
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd", segments=["segment"]),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20),
    )
    assert any(
        check.name.endswith("segment_cardinality.segment") and check.status == Status.WARNING
        for check in checks
    )


def test_small_segment_warns(sample_frame: pl.DataFrame) -> None:
    frame = sample_frame.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.lit("tiny"))
        .otherwise(pl.lit("regular"))
        .alias("segment")
    )
    checks = validate_contract(
        frame,
        ColumnConfig(target="target", pd="pd", segments=["segment"]),
        ValidationOptions(min_events=5, min_non_events=5, min_rows=20, min_segment_size=5),
    )
    assert any(
        check.name.endswith("segment_size.segment") and check.status == Status.WARNING
        for check in checks
    )
