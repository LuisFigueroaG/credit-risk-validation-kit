from typing import cast

import pytest

from credit_risk_validation.metrics.calibration import calibration_metrics


def test_brier_and_log_loss_are_calculated() -> None:
    metrics, table = calibration_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=2)
    assert metrics["brier"].value == pytest.approx(0.025)
    assert metrics["log_loss"].value is not None
    assert table.height == 2


def test_ece_manual() -> None:
    metrics, _ = calibration_metrics([0, 1, 0, 1], [0.25, 0.25, 0.75, 0.75], n_bins=2)
    assert metrics["ece"].value == pytest.approx(0.25)


def test_oe_ratio() -> None:
    metrics, _ = calibration_metrics([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], n_bins=2)
    assert metrics["oe_ratio"].value == pytest.approx(1.0)


def test_calibration_intercept_and_slope_are_reported() -> None:
    metrics, _ = calibration_metrics(
        [0, 0, 0, 1, 1, 1],
        [0.05, 0.10, 0.20, 0.65, 0.80, 0.90],
        n_bins=3,
    )
    assert metrics["calibration_intercept"].value is not None
    assert metrics["calibration_slope"].value is not None


def test_calibration_intercept_and_slope_are_insufficient_for_single_class() -> None:
    metrics, table = calibration_metrics([0, 0, 0], [0.1, 0.2, 0.3], n_bins=2)
    assert table.is_empty()
    assert metrics["calibration_intercept"].status.value == "INSUFFICIENT_DATA"
    assert metrics["calibration_intercept"].value is None
    assert metrics["calibration_slope"].status.value == "INSUFFICIENT_DATA"
    assert metrics["calibration_slope"].value is None


def test_calibration_table_contains_audit_fields() -> None:
    _, table = calibration_metrics([0, 1, 0, 1], [0.2, 0.4, 0.6, 0.8], n_bins=2)

    expected_columns = {
        "expected_defaults",
        "observed_defaults",
        "event_rate",
        "avg_pd",
        "rel_error",
        "oe_ratio",
        "lower_event_rate",
        "upper_event_rate",
        "status",
    }
    assert expected_columns.issubset(set(table.columns))
    assert table["expected_defaults"].sum() == pytest.approx(2.0)
    assert table["observed_defaults"].sum() == pytest.approx(2.0)
    lower_min = cast(float, table["lower_event_rate"].min())
    upper_max = cast(float, table["upper_event_rate"].max())
    assert lower_min >= 0
    assert upper_max <= 1
