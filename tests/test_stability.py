import polars as pl
import pytest

from credit_risk_validation import PDValidationSuite
from credit_risk_validation.metrics.stability import psi_categorical, psi_numeric


def test_numeric_psi_zero_for_same_distribution() -> None:
    value, table = psi_numeric([0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], n_bins=2)
    assert value == pytest.approx(0.0)
    assert table.height > 0


def test_numeric_psi_positive_for_shift() -> None:
    value, _ = psi_numeric([0.1, 0.2, 0.3, 0.4], [0.7, 0.8, 0.9, 1.0], n_bins=2)
    assert value > 0


def test_categorical_psi_handles_new_category() -> None:
    value, table = psi_categorical(["A", "A", "B"], ["A", "C", "C"])
    assert value > 0
    assert "OTHER" in table["category"].to_list()


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
