import pytest

from credit_risk_validation.metrics.discrimination import auc_gini_ks


def test_perfect_auc_is_one() -> None:
    metrics = auc_gini_ks([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["auc"].value == pytest.approx(1.0)
    assert metrics["gini"].value == pytest.approx(1.0)


def test_inverted_auc_is_zero() -> None:
    metrics = auc_gini_ks([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert metrics["auc"].value == pytest.approx(0.0)
    assert metrics["gini"].value == pytest.approx(-1.0)


def test_ks_manual_small_vector() -> None:
    metrics = auc_gini_ks([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["ks"].value == pytest.approx(1.0)
