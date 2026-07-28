import polars as pl
import pytest
from pydantic import ValidationError

from credit_risk_validation.config import ValidationOptions
from credit_risk_validation.metrics.discrimination import (
    auc_gini_ks,
    discrimination_from_frame,
    lift_table,
)


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


def test_constant_score_has_zero_ks_and_one_lift_bin() -> None:
    target = [0, 1, 0, 1]
    score = [0.5, 0.5, 0.5, 0.5]

    metrics = auc_gini_ks(target, score)
    table = lift_table(target, score, n_bins=10)

    assert metrics["auc"].value == pytest.approx(0.5)
    assert metrics["ks"].value == pytest.approx(0.0)
    assert table.height == 1
    assert table["count"].to_list() == [4]
    assert table["min_score"].to_list() == [0.5]
    assert table["max_score"].to_list() == [0.5]


def test_weighted_ks_is_evaluated_at_tie_boundaries() -> None:
    metrics = auc_gini_ks(
        [1, 0, 1, 0],
        [0.9, 0.9, 0.5, 0.5],
        sample_weight=[3.0, 1.0, 1.0, 3.0],
    )

    assert metrics["ks"].value == pytest.approx(0.5)


def test_lift_table_keeps_partial_ties_in_the_same_bin() -> None:
    table = lift_table(
        [1, 0, 1, 0, 1, 0, 0, 1],
        [0.9, 0.9, 0.8, 0.7, 0.7, 0.7, 0.2, 0.1],
        n_bins=4,
        sample_weight=[2.0, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 4.0],
    )

    assert table.height == 3
    assert table["count"].to_list() == [2, 4, 2]
    assert table["weighted_count"].to_list() == [3.0, 7.0, 5.0]
    assert table["min_score"].to_list() == [0.9, 0.7, 0.1]
    assert table["max_score"].to_list() == [0.9, 0.8, 0.2]


def test_discrimination_is_invariant_to_input_order_with_weighted_ties() -> None:
    target = [1, 0, 1, 0, 1, 0]
    score = [0.9, 0.9, 0.6, 0.6, 0.2, 0.2]
    weight = [3.0, 1.0, 2.0, 4.0, 1.0, 5.0]
    permutation = [5, 2, 0, 4, 1, 3]

    expected_metrics = auc_gini_ks(target, score, sample_weight=weight)
    expected_table = lift_table(target, score, n_bins=4, sample_weight=weight)
    actual_metrics = auc_gini_ks(
        [target[index] for index in permutation],
        [score[index] for index in permutation],
        sample_weight=[weight[index] for index in permutation],
    )
    actual_table = lift_table(
        [target[index] for index in permutation],
        [score[index] for index in permutation],
        n_bins=4,
        sample_weight=[weight[index] for index in permutation],
    )

    for metric_name in ("auc", "gini", "ks"):
        assert actual_metrics[metric_name].value == pytest.approx(
            expected_metrics[metric_name].value
        )
    assert actual_table.to_dicts() == expected_table.to_dicts()


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


def test_lower_is_riskier_preserves_original_score_ranges() -> None:
    frame = pl.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "score": [900.0, 800.0, 200.0, 100.0],
        }
    )

    metrics, table = discrimination_from_frame(
        frame,
        target_col="target",
        score_col="score",
        weight_col=None,
        validation=ValidationOptions(
            score_direction="lower_is_riskier",
            n_bins=2,
            min_events=1,
            min_non_events=1,
        ),
    )

    assert metrics["auc"].value == pytest.approx(1.0)
    assert table["min_score"].to_list() == [100.0, 800.0]
    assert table["max_score"].to_list() == [200.0, 900.0]


def test_score_direction_rejects_unsupported_value() -> None:
    with pytest.raises(ValidationError):
        ValidationOptions(score_direction="sideways")
