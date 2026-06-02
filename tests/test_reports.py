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
    assert "Reference Value" in rendered
    assert "Current Value" in rendered
    assert "Delta" in rendered
    assert "Warning Threshold" in rendered


def test_html_report_respects_optional_report_sections(
    sample_frame: pl.DataFrame, tmp_path: Path
) -> None:
    suite = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        segment_cols=["segment"],
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    )
    suite.config.report.include_charts = False
    suite.config.report.include_model_card = False
    suite.config.report.include_methodology = False
    result = suite.run(reference_data=sample_frame, current_data=sample_frame)

    report_path = tmp_path / "report.html"
    result.to_html(report_path)
    rendered = report_path.read_text(encoding="utf-8")

    assert "Plotly.newPlot" not in rendered
    assert "Model Card" not in rendered
    assert "Methodology Appendix" not in rendered


def test_html_report_supports_spanish_language(sample_frame: pl.DataFrame, tmp_path: Path) -> None:
    suite = PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        segment_cols=["segment"],
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    )
    suite.config.report.language = "es"
    suite.config.report.title = "Reporte de validacion PD"
    result = suite.run(reference_data=sample_frame, current_data=sample_frame)

    report_path = tmp_path / "report.html"
    result.to_html(report_path)
    rendered = report_path.read_text(encoding="utf-8")

    assert '<html lang="es">' in rendered
    assert "Resumen ejecutivo" in rendered
    assert "Calidad de datos" in rendered
    assert "No certifica cumplimiento regulatorio" in rendered
    assert "Lift y captura de eventos" in rendered
    assert "Valor reference" in rendered
    assert "Valor current" in rendered
    assert "Umbral warning" in rendered
