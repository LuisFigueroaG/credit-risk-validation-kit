from collections.abc import Sequence

import polars as pl
import pytest

from credit_risk_validation import PDValidationSuite
from credit_risk_validation.status import Status


def _comparison_frame(
    *,
    include_score: bool = True,
    event_pd: float = 0.8,
    non_event_pd: float = 0.2,
    score_opposes_pd: bool = False,
) -> pl.DataFrame:
    target = [0, 1] * 40
    pd_values = [event_pd if value else non_event_pd for value in target]
    payload: dict[str, Sequence[object]] = {
        "target": target,
        "pd": pd_values,
        "segment": ["A", "A", "B", "B"] * 20,
    }
    if include_score:
        payload["score"] = [1.0 - value for value in pd_values] if score_opposes_pd else pd_values
    return pl.DataFrame(payload)


def _suite(*, score_direction: str = "lower_is_riskier") -> PDValidationSuite:
    return PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        segment_cols=["segment"],
        n_bins=2,
        min_events=1,
        min_non_events=1,
        min_rows=1,
        score_direction=score_direction,
    )


@pytest.mark.parametrize("missing_score_from", ["reference", "current"])
def test_drift_falls_back_to_pd_for_both_datasets_without_mutating_config(
    missing_score_from: str,
) -> None:
    reference = _comparison_frame(include_score=missing_score_from != "reference")
    current = _comparison_frame(include_score=missing_score_from != "current")
    suite = _suite(score_direction="lower_is_riskier")

    result = suite.run_drift(reference_data=reference, current_data=current)

    assert result.metrics["auc"].reference_value == pytest.approx(1.0)
    assert result.metrics["auc"].current_value == pytest.approx(1.0)
    assert suite.config.columns.score == "score"
    assert suite.config.validation.score_direction == "lower_is_riskier"
    assert result.metrics["psi_score"].status == Status.NOT_APPLICABLE

    for table_name in ["segment_analysis", "current_segment_analysis"]:
        table = result.tables[table_name]
        assert table["auc"].to_list() == pytest.approx([1.0, 1.0])


def test_drift_uses_configured_score_when_it_exists_in_both_datasets() -> None:
    reference = _comparison_frame(score_opposes_pd=True)
    current = _comparison_frame(score_opposes_pd=True)

    result = _suite(score_direction="lower_is_riskier").run_drift(
        reference_data=reference,
        current_data=current,
    )

    assert result.metrics["auc"].reference_value == pytest.approx(1.0)
    assert result.metrics["auc"].current_value == pytest.approx(1.0)
    assert result.metrics["psi_score"].status != Status.NOT_APPLICABLE


def test_drift_exposes_current_lift_and_calibration_tables() -> None:
    reference = _comparison_frame(event_pd=0.8, non_event_pd=0.2, score_opposes_pd=True)
    current = _comparison_frame(event_pd=0.6, non_event_pd=0.4, score_opposes_pd=True)

    result = _suite().run_drift(reference_data=reference, current_data=current)

    reference_lift = result.tables["lift_table"]
    current_lift = result.tables["current_lift_table"]
    reference_calibration = result.tables["calibration_bins"]
    current_calibration = result.tables["current_calibration_bins"]

    assert current_lift.height > 0
    assert current_calibration.height > 0
    assert current_lift["count"].sum() == current.height
    assert current_calibration["count"].sum() == current.height
    assert current_lift["max_score"].max() != reference_lift["max_score"].max()
    assert current_calibration["max_pd"].max() != reference_calibration["max_pd"].max()
