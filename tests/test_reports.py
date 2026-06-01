import polars as pl

from credit_risk_validation import PDValidationSuite


def test_html_contains_key_sections(sample_frame: pl.DataFrame) -> None:
    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    ).run(reference_data=sample_frame)
    html = result.to_dict()
    assert html["status"]
    rendered = result.to_dict()["disclaimer"]
    assert "does not approve" in rendered
