import pytest

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
    assert "C" in table["category"].to_list()
