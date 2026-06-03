import polars as pl

from credit_risk_validation import PDValidationSuite
from credit_risk_validation.status import Status


def test_segment_analysis_returns_rows(sample_frame: pl.DataFrame) -> None:
    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        segment_cols=["segment"],
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    ).run(validation_data=sample_frame)
    table = result.tables["segment_analysis"]
    assert table.height == 2
    assert set(table["status"].to_list()) == {Status.OK.value}


def test_small_segment_is_insufficient(sample_frame: pl.DataFrame) -> None:
    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        segment_cols=["segment"],
        min_events=100,
        min_non_events=100,
        min_rows=20,
        score_direction="lower_is_riskier",
    ).run(validation_data=sample_frame)
    assert Status.INSUFFICIENT_DATA.value in result.tables["segment_analysis"]["status"].to_list()
