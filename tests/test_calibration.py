from typing import cast
from warnings import warn

import pytest
from sklearn.exceptions import ConvergenceWarning

from credit_risk_validation.config import ThresholdConfig
from credit_risk_validation.metrics.calibration import calibration_metrics
from credit_risk_validation.status import Status


def test_brier_and_log_loss_are_calculated() -> None:
    metrics, table = calibration_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=2)
    assert metrics["brier"].value == pytest.approx(0.025)
    assert metrics["log_loss"].value is not None
    assert table.height == 2


def test_ece_manual() -> None:
    metrics, _ = calibration_metrics([0, 1, 0, 1], [0.25, 0.25, 0.75, 0.75], n_bins=2)
    assert metrics["ece"].value == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("threshold", "expected_status"),
    [
        (ThresholdConfig(warning=0.26, critical=0.50), Status.OK),
        (ThresholdConfig(warning=0.25, critical=0.50), Status.WARNING),
        (ThresholdConfig(warning=0.10, critical=0.25), Status.CRITICAL),
    ],
)
def test_ece_and_mce_thresholds_are_inclusive(
    threshold: ThresholdConfig, expected_status: Status
) -> None:
    metrics, _ = calibration_metrics(
        [0, 1, 0, 1],
        [0.25, 0.25, 0.75, 0.75],
        n_bins=2,
        calibration_abs_error_threshold=threshold,
    )

    assert metrics["ece"].value == pytest.approx(0.25)
    assert metrics["mce"].value == pytest.approx(0.25)
    assert metrics["ece"].status == expected_status
    assert metrics["mce"].status == expected_status


def test_default_calibration_thresholds_are_reported() -> None:
    metrics, _ = calibration_metrics([0, 1], [0.25, 0.75], n_bins=2)

    assert metrics["ece"].threshold_warning == pytest.approx(0.02)
    assert metrics["ece"].threshold_critical == pytest.approx(0.05)
    assert metrics["mce"].threshold_warning == pytest.approx(0.02)
    assert metrics["mce"].threshold_critical == pytest.approx(0.05)


def test_oe_ratio() -> None:
    metrics, _ = calibration_metrics([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], n_bins=2)
    assert metrics["oe_ratio"].value == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("y_true", "pd_values", "expected_value", "expected_status"),
    [
        ([1, 0, 0, 0], [0.5] * 4, 0.5, Status.CRITICAL),
        ([1, 0, 0, 0], [1 / 3] * 4, 0.75, Status.WARNING),
        ([1, 1, 0, 0], [0.4] * 4, 1.25, Status.WARNING),
        ([1, 1, 0, 0], [0.25] * 4, 2.0, Status.CRITICAL),
        ([1, 1, 0, 0], [0.5] * 4, 1.0, Status.OK),
    ],
)
def test_oe_ratio_uses_inclusive_two_sided_thresholds(
    y_true: list[int],
    pd_values: list[float],
    expected_value: float,
    expected_status: Status,
) -> None:
    threshold = ThresholdConfig(
        warning_low=0.75,
        warning_high=1.25,
        critical_low=0.5,
        critical_high=2.0,
    )

    metrics, _ = calibration_metrics(
        y_true,
        pd_values,
        n_bins=2,
        oe_ratio_threshold=threshold,
    )

    metric = metrics["oe_ratio"]
    assert metric.value == pytest.approx(expected_value)
    assert metric.status == expected_status
    assert metric.threshold_warning_low == pytest.approx(0.75)
    assert metric.threshold_warning_high == pytest.approx(1.25)
    assert metric.threshold_critical_low == pytest.approx(0.5)
    assert metric.threshold_critical_high == pytest.approx(2.0)


def test_calibration_intercept_and_slope_are_unregularized() -> None:
    y_true = [1, 1, *([0] * 8), *([1] * 8), 0, 0]
    pd_values = [0.2] * 10 + [0.8] * 10

    metrics, _ = calibration_metrics(
        y_true,
        pd_values,
        n_bins=2,
    )

    assert metrics["calibration_intercept"].status == Status.OK
    assert metrics["calibration_intercept"].value == pytest.approx(0.0, abs=1e-6)
    assert metrics["calibration_slope"].status == Status.OK
    assert metrics["calibration_slope"].value == pytest.approx(1.0, abs=1e-6)


def test_calibration_intercept_and_slope_are_insufficient_under_separation() -> None:
    metrics, _ = calibration_metrics(
        [0, 0, 0, 1, 1, 1],
        [0.05, 0.10, 0.20, 0.65, 0.80, 0.90],
        n_bins=3,
    )

    for name in ["calibration_intercept", "calibration_slope"]:
        assert metrics[name].value is None
        assert metrics[name].status == Status.INSUFFICIENT_DATA
        assert "separation" in metrics[name].message


def test_calibration_intercept_and_slope_are_insufficient_for_constant_pd() -> None:
    metrics, _ = calibration_metrics([0, 1, 0, 1], [0.5] * 4, n_bins=2)

    for name in ["calibration_intercept", "calibration_slope"]:
        assert metrics[name].value is None
        assert metrics[name].status == Status.INSUFFICIENT_DATA
        assert "constant PD" in metrics[name].message


def test_calibration_fit_reports_non_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def warn_on_fit(*args: object, **kwargs: object) -> None:
        warn("did not converge", ConvergenceWarning, stacklevel=2)

    monkeypatch.setattr(
        "credit_risk_validation.metrics.calibration.LogisticRegression.fit",
        warn_on_fit,
    )
    y_true = [1, 1, *([0] * 8), *([1] * 8), 0, 0]
    pd_values = [0.2] * 10 + [0.8] * 10

    metrics, _ = calibration_metrics(y_true, pd_values, n_bins=2)

    for name in ["calibration_intercept", "calibration_slope"]:
        assert metrics[name].value is None
        assert metrics[name].status == Status.INSUFFICIENT_DATA
        assert "did not converge" in metrics[name].message


def test_calibration_intercept_and_slope_are_insufficient_for_single_class() -> None:
    oe_threshold = ThresholdConfig(
        warning_low=0.8,
        warning_high=1.2,
        critical_low=0.7,
        critical_high=1.3,
    )
    metrics, table = calibration_metrics(
        [0, 0, 0],
        [0.1, 0.2, 0.3],
        n_bins=2,
        oe_ratio_threshold=oe_threshold,
    )
    assert table.is_empty()
    assert metrics["calibration_intercept"].status.value == "INSUFFICIENT_DATA"
    assert metrics["calibration_intercept"].value is None
    assert metrics["calibration_slope"].status.value == "INSUFFICIENT_DATA"
    assert metrics["calibration_slope"].value is None
    assert metrics["ece"].threshold_warning == pytest.approx(0.02)
    assert metrics["ece"].threshold_critical == pytest.approx(0.05)
    assert metrics["oe_ratio"].threshold_warning_low == pytest.approx(0.8)
    assert metrics["oe_ratio"].threshold_warning_high == pytest.approx(1.2)
    assert metrics["oe_ratio"].threshold_critical_low == pytest.approx(0.7)
    assert metrics["oe_ratio"].threshold_critical_high == pytest.approx(1.3)


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
        "message",
    }
    assert expected_columns.issubset(set(table.columns))
    assert table["expected_defaults"].sum() == pytest.approx(2.0)
    assert table["observed_defaults"].sum() == pytest.approx(2.0)
    lower_min = cast(float, table["lower_event_rate"].min())
    upper_max = cast(float, table["upper_event_rate"].max())
    assert lower_min >= 0
    assert upper_max <= 1


def test_calibration_bins_keep_worst_status_when_degenerate() -> None:
    _, table = calibration_metrics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        n_bins=2,
    )

    assert set(table["status"].to_list()) == {"CRITICAL"}
    messages = " ".join(table["message"].to_list())
    assert "Calibration bin has no events" in messages
    assert "Calibration bin has no non-events" in messages
    assert "critical threshold" in messages


def test_calibration_bins_warn_when_degenerate_but_within_error_thresholds() -> None:
    _, table = calibration_metrics(
        [0, 0, 1, 1],
        [0.001, 0.002, 0.998, 0.999],
        n_bins=2,
    )

    assert set(table["status"].to_list()) == {"WARNING"}
