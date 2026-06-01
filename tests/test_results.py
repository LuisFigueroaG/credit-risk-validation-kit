from pathlib import Path

import polars as pl

from credit_risk_validation import PDValidationSuite, __version__


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
    payload = result.to_dict()
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    model_card = (tmp_path / "model_card.md").read_text(encoding="utf-8")

    assert (tmp_path / "metrics.json").exists()
    assert payload["version"] == __version__
    assert payload["metadata"]["library_version"] == __version__
    assert len(payload["metadata"]["config_sha256"]) == 64
    assert len(payload["metadata"]["reference_schema_sha256"]) == 64
    assert "PD Model Validation Report" in html
    assert "Config SHA-256" in html
    assert (tmp_path / "tables" / "lift_table.csv").exists()
    assert (tmp_path / "model_card.md").exists()
    assert "Config SHA-256" in model_card
