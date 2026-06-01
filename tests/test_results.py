from pathlib import Path

import polars as pl

from credit_risk_validation import PDValidationSuite


def test_result_exports(sample_frame: pl.DataFrame, tmp_path: Path) -> None:
    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    ).run(reference_data=sample_frame, current_data=sample_frame)
    result.to_json(tmp_path / "metrics.json")
    result.to_html(tmp_path / "report.html")
    result.to_tables(tmp_path / "tables")
    result.to_model_card(tmp_path / "model_card.md")
    assert (tmp_path / "metrics.json").exists()
    assert "PD Model Validation Report" in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert (tmp_path / "tables" / "lift_table.csv").exists()
    assert (tmp_path / "model_card.md").exists()
