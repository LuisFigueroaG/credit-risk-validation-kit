import polars as pl
import pytest
from pydantic import ValidationError

from credit_risk_validation.config import ValidationOptions
from credit_risk_validation.metrics.discrimination import auc_gini_ks, discrimination_from_frame


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


def test_higher_is_safer_score_direction_is_inverted() -> None:
    frame = pl.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "score": [900, 800, 300, 200],
        }
    )
    metrics, _ = discrimination_from_frame(
        frame,
        target_col="target",
        score_col="score",
        weight_col=None,
        validation=ValidationOptions(
            score_direction="higher_is_safer", min_events=1, min_non_events=1
        ),
    )
    assert metrics["auc"].value == pytest.approx(1.0)


def test_score_direction_rejects_unsupported_value() -> None:
    with pytest.raises(ValidationError):
        ValidationOptions(score_direction="sideways")
