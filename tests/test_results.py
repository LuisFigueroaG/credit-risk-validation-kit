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


def test_model_card_supports_spanish_language(sample_frame: pl.DataFrame, tmp_path: Path) -> None:
    suite = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    )
    suite.config.report.language = "es"
    result = suite.run(reference_data=sample_frame, current_data=sample_frame)

    result.to_model_card(tmp_path / "model_card.md")
    model_card = (tmp_path / "model_card.md").read_text(encoding="utf-8")

    assert "## Informacion del modelo" in model_card
    assert "## Estado de validacion" in model_card
    assert "## Uso previsto" in model_card
    assert "No certifica cumplimiento regulatorio" in model_card


def test_metric_results_include_audit_fields(sample_frame: pl.DataFrame, tmp_path: Path) -> None:
    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    ).run(reference_data=sample_frame, current_data=sample_frame)
    result.to_tables(tmp_path / "tables")

    auc = result.to_dict()["metrics"]["auc"]
    for field in [
        "reference_value",
        "current_value",
        "delta",
        "threshold_warning",
        "threshold_critical",
        "sample_size",
        "event_count",
        "non_event_count",
    ]:
        assert field in auc
    assert auc["sample_size"] == sample_frame.height
    assert auc["event_count"] == int(sample_frame["target"].sum())
    assert auc["non_event_count"] == sample_frame.height - int(sample_frame["target"].sum())

    discrimination_table = pl.read_csv(tmp_path / "tables" / "discrimination.csv")
    assert "sample_size" in discrimination_table.columns
    assert "event_count" in discrimination_table.columns


def test_empty_optional_tables_are_exported_with_headers(
    sample_frame: pl.DataFrame, tmp_path: Path
) -> None:
    result = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    ).run(reference_data=sample_frame)

    result.to_tables(tmp_path / "tables")

    for table_name, expected_columns in {
        "psi_by_variable.csv": {"variable", "psi", "status"},
        "segment_metrics.csv": {"segment", "status", "auc"},
        "temporal_metrics.csv": {"dataset", "period", "status"},
    }.items():
        table_path = tmp_path / "tables" / table_name
        assert table_path.exists()
        exported = pl.read_csv(table_path)
        assert expected_columns.issubset(set(exported.columns))
