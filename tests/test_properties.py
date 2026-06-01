from hypothesis import given, settings
from hypothesis import strategies as st

from credit_risk_validation.metrics.calibration import calibration_metrics
from credit_risk_validation.metrics.discrimination import auc_gini_ks
from credit_risk_validation.metrics.stability import psi_numeric


@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=2, max_size=50))
@settings(deadline=None)
def test_brier_is_non_negative(probabilities: list[float]) -> None:
    y_true = [0, 1] * ((len(probabilities) + 1) // 2)
    metrics, _ = calibration_metrics(y_true[: len(probabilities)], probabilities, n_bins=2)
    if metrics["brier"].value is not None:
        assert metrics["brier"].value >= 0


@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=4, max_size=50))
@settings(deadline=None)
def test_auc_in_unit_interval(scores: list[float]) -> None:
    y_true = [0, 1] * ((len(scores) + 1) // 2)
    metrics = auc_gini_ks(y_true[: len(scores)], scores)
    if metrics["auc"].value is not None:
        assert 0 <= metrics["auc"].value <= 1


@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=4, max_size=50))
@settings(deadline=None)
def test_psi_is_non_negative(values: list[float]) -> None:
    value, _ = psi_numeric(values, values, n_bins=4)
    assert value >= -1e-12
