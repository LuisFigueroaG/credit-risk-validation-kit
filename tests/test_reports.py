from pathlib import Path

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


def test_html_report_includes_embedded_charts(sample_frame: pl.DataFrame, tmp_path: Path) -> None:
    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        segment_cols=["segment"],
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    ).run(reference_data=sample_frame, current_data=sample_frame)

    report_path = tmp_path / "report.html"
    result.to_html(report_path)
    rendered = report_path.read_text(encoding="utf-8")

    assert "Plotly.newPlot" in rendered
    assert "Lift and Event Capture" in rendered
    assert "PD PSI by Bin" in rendered
    assert "Segment Calibration Summary" in rendered
