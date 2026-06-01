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
