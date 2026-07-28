import hashlib

import numpy as np
import polars as pl
import pytest

from credit_risk_validation import PDValidationSuite
from credit_risk_validation.config import ThresholdConfig
from credit_risk_validation.metrics.stability import (
    psi_categorical,
    psi_for_column,
    psi_numeric,
    stability_from_frames,
)


def test_numeric_psi_zero_for_same_distribution() -> None:
    value, table = psi_numeric([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], n_bins=2)
    assert value == pytest.approx(0.0)
    assert table.height > 0


def test_numeric_psi_positive_for_shift() -> None:
    value, _ = psi_numeric([0.1, 0.2, 0.3, 0.4], [0.7, 0.8, 0.9, 1.0], n_bins=2)
    assert value > 0


def test_numeric_psi_preserves_missing_and_non_finite_values() -> None:
    value, table = psi_numeric(
        [1.0, 2.0, 1.0, 2.0],
        [1.0, 2.0, None, float("nan")],
        n_bins=2,
    )

    missing = table.filter(pl.col("bin") == 0).to_dicts()[0]
    assert value > 0
    assert missing["bucket"] == "MISSING"
    assert missing["reference_count"] == 0
    assert missing["current_count"] == 2
    assert table["reference_share"].sum() == pytest.approx(1.0)
    assert table["current_share"].sum() == pytest.approx(1.0)


def test_numeric_psi_all_missing_samples_are_stable() -> None:
    value, table = psi_numeric(
        [None, float("nan")],
        [float("inf"), float("-inf")],
        n_bins=2,
    )

    assert value == pytest.approx(0.0)
    assert table.to_dicts() == [
        {
            "bin": 0,
            "bucket": "MISSING",
            "reference_count": 2,
            "current_count": 2,
            "reference_share": 1.0,
            "current_share": 1.0,
            "psi": 0.0,
        }
    ]


def test_numeric_psi_detects_all_missing_to_finite_shift() -> None:
    value, table = psi_numeric([None, np.nan], [1.0, 2.0], n_bins=2)

    assert value > 0
    assert set(table["bin"].to_list()) == {0, 1}
    assert table["reference_share"].sum() == pytest.approx(1.0)
    assert table["current_share"].sum() == pytest.approx(1.0)


def test_numeric_psi_for_column_keeps_nulls_in_denominator() -> None:
    reference = pl.DataFrame({"numeric_segment": [1.0, 2.0, 1.0, 2.0]})
    current = pl.DataFrame({"numeric_segment": [1.0, 2.0, None, None]})

    value, table, variable_type = psi_for_column(
        reference,
        current,
        column="numeric_segment",
        n_bins=2,
    )

    assert value > 0
    assert variable_type == "numeric"
    assert table.filter(pl.col("bin") == 0)["current_share"].item() == pytest.approx(0.5)


def test_categorical_psi_handles_new_category() -> None:
    value, table = psi_categorical(["A", "A", "B"], ["A", "C", "C"])
    assert value > 0
    assert "OTHER" in table["category"].to_list()


def test_psi_metric_names_reserve_pd_and_score_keys() -> None:
    reference = pl.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "pd": [0.1, 0.9, 0.1, 0.9],
            "score": [10.0, 90.0, 10.0, 90.0],
            "PD!": ["A", "A", "B", "B"],
            "Score!": ["A", "A", "B", "B"],
        }
    )
    current = reference.with_columns(
        pl.lit("C").alias("PD!"),
        pl.lit("D").alias("Score!"),
    )

    metrics, tables = stability_from_frames(
        reference,
        current,
        pd_col="pd",
        score_col="score",
        target_col="target",
        variable_cols=["PD!", "Score!"],
        n_bins=2,
        psi_threshold=ThresholdConfig(warning=0.1, critical=0.25),
    )

    pd_key = f"psi_pd_{hashlib.sha256(b'PD!').hexdigest()[:8]}"
    score_key = f"psi_score_{hashlib.sha256(b'Score!').hexdigest()[:8]}"
    custom_pd_value = metrics[pd_key].value
    custom_score_value = metrics[score_key].value
    assert metrics["psi_pd"].value == pytest.approx(0.0)
    assert metrics["psi_score"].value == pytest.approx(0.0)
    assert custom_pd_value is not None
    assert custom_pd_value > 0
    assert custom_score_value is not None
    assert custom_score_value > 0
    assert tables["psi_pd"]["variable"].unique().to_list() == ["pd"]
    assert tables["psi_score"]["variable"].unique().to_list() == ["score"]
    assert tables[pd_key]["variable"].unique().to_list() == ["PD!"]
    assert tables[score_key]["variable"].unique().to_list() == ["Score!"]


def test_colliding_custom_psi_names_are_stable_across_column_order() -> None:
    reference = pl.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "pd": [0.1, 0.9, 0.1, 0.9],
            "A-B": ["A", "A", "B", "B"],
            "A B": ["A", "B", "A", "B"],
        }
    )
    current = reference.with_columns(
        pl.lit("C").alias("A-B"),
        pl.lit("D").alias("A B"),
    )
    expected_custom_keys = {
        f"psi_a_b_{hashlib.sha256(column.encode()).hexdigest()[:8]}" for column in ["A-B", "A B"]
    }

    generated_key_sets = []
    for variable_cols in [["A-B", "A B"], ["A B", "A-B"]]:
        metrics, tables = stability_from_frames(
            reference,
            current,
            pd_col="pd",
            score_col=None,
            target_col="target",
            variable_cols=variable_cols,
            n_bins=2,
            psi_threshold=ThresholdConfig(warning=0.1, critical=0.25),
        )
        custom_keys = {key for key in metrics if key.startswith("psi_a_b_")}
        generated_key_sets.append(custom_keys)
        assert custom_keys == expected_custom_keys
        assert custom_keys.issubset(tables)

    assert generated_key_sets[0] == generated_key_sets[1]


def test_non_colliding_custom_psi_name_remains_compatible() -> None:
    reference = pl.DataFrame({"target": [0, 1], "pd": [0.1, 0.9], "segment": ["A", "B"]})

    metrics, tables = stability_from_frames(
        reference,
        reference,
        pd_col="pd",
        score_col=None,
        target_col="target",
        variable_cols=["segment"],
        n_bins=2,
        psi_threshold=ThresholdConfig(warning=0.1, critical=0.25),
    )

    assert "psi_segment" in metrics
    assert "psi_segment" in tables


def test_temporal_metrics_include_performance_metrics() -> None:
    frame = pl.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "pd": [0.05, 0.65, 0.10, 0.75, 0.08, 0.70, 0.12, 0.80],
            "period": [
                "2025-01",
                "2025-01",
                "2025-01",
                "2025-01",
                "2025-02",
                "2025-02",
                "2025-02",
                "2025-02",
            ],
        }
    )
    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col=None,
        period_col="period",
        n_bins=2,
        min_events=1,
        min_non_events=1,
        min_rows=4,
    ).run(validation_data=frame)

    table = result.tables["temporal_metrics"]
    assert table.height == 2
    for column in ["auc", "gini", "ks", "brier", "log_loss", "ece", "oe_ratio", "status"]:
        assert column in table.columns
    assert table["auc"].drop_nulls().len() == 2
    assert set(table["status"].to_list()) == {"OK"}


def test_temporal_metrics_mark_single_class_periods_insufficient() -> None:
    frame = pl.DataFrame(
        {
            "target": [0, 0, 0, 0, 0, 1, 0, 1],
            "pd": [0.05, 0.07, 0.10, 0.12, 0.08, 0.70, 0.12, 0.80],
            "period": [
                "2025-01",
                "2025-01",
                "2025-01",
                "2025-01",
                "2025-02",
                "2025-02",
                "2025-02",
                "2025-02",
            ],
        }
    )
    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col=None,
        period_col="period",
        n_bins=2,
        min_events=1,
        min_non_events=1,
        min_rows=4,
    ).run(validation_data=frame)

    table = result.tables["temporal_metrics"]
    insufficient = table.filter(pl.col("period") == "2025-01").to_dicts()[0]
    assert insufficient["status"] == "INSUFFICIENT_DATA"
    assert insufficient["auc"] is None


def test_stability_includes_variable_and_segment_drift_tables(
    sample_frame: pl.DataFrame,
) -> None:
    current = sample_frame.with_columns(
        pl.when(pl.arange(0, pl.len()) < 40)
        .then(pl.lit("C"))
        .otherwise(pl.col("segment"))
        .alias("segment")
    )

    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        segment_cols=["segment"],
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    ).run_drift(reference_data=sample_frame, current_data=current)

    psi_by_variable = result.tables["psi_by_variable"]
    csi_by_variable = result.tables["csi_by_variable"]
    segment_drift = result.tables["segment_drift"]

    assert set(psi_by_variable["variable"].to_list()) == {"pd", "score", "segment"}
    assert csi_by_variable["variable"].to_list() == ["segment"]
    assert "OTHER" in segment_drift["segment"].to_list()
    assert "bad_rate_delta" in segment_drift.columns
    assert "avg_pd_delta" in segment_drift.columns
