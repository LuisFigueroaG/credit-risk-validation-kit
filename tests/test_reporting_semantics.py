from dataclasses import replace

import polars as pl

from credit_risk_validation import PDValidationSuite
from credit_risk_validation.reports.html import (
    _bad_rate_chart,
    _calibration_chart,
    _format_cell,
    _kpi_card,
    _lift_chart,
    _segment_drift_panel,
    _table_cell,
    _threshold_legend,
    render_html_report,
)
from credit_risk_validation.reports.i18n import translations
from credit_risk_validation.reports.model_card import render_model_card
from credit_risk_validation.schemas import MetricResult
from credit_risk_validation.status import Status


def _suite(*, segments: list[str] | None = None) -> PDValidationSuite:
    return PDValidationSuite(
        target_col="target",
        pd_col="pd",
        score_col="score",
        segment_cols=segments,
        min_events=5,
        min_non_events=5,
        min_rows=20,
        score_direction="lower_is_riskier",
    )


def test_oe_ratio_is_formatted_as_ratio_not_percentage() -> None:
    assert _format_cell(1.18, header="oe_ratio") == "1.18"
    assert _format_cell(0.18, header="observed_rate") == "18.00%"


def test_status_cell_rejects_attribute_injection() -> None:
    rendered = _table_cell("status", 'OK"\nonmouseover="globalThis.PWNED=1')
    opening_tag = rendered.split(">", maxsplit=1)[0]

    assert opening_tag == '<td class="status-cell unknown"'
    assert "onmouseover" not in opening_tag
    assert "&quot;" in rendered


def test_model_card_neutralizes_html_and_markdown(sample_frame: pl.DataFrame) -> None:
    result = _suite().run(validation_data=sample_frame)
    result.config.model.name = "<script>globalThis.PWNED=1</script>\n## injected"
    result.config.model.owner = "[owner](javascript:alert(1))"
    result.config.model.purpose = '<img src=x onerror="alert(1)">'
    result.metrics["unsafe"] = MetricResult('<svg onload="alert(1)">', 1.0)

    rendered = render_model_card(result)

    assert "<script>" not in rendered
    assert "<img" not in rendered
    assert "<svg" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "\\[owner\\]\\(javascript:alert\\(1\\)\\)" in rendered
    assert "\n## injected" not in rendered


def test_model_card_has_own_navigation_without_methodology(sample_frame: pl.DataFrame) -> None:
    suite = _suite()
    suite.config.report.include_methodology = False
    suite.config.report.include_model_card = True
    suite.config.report.include_charts = False
    result = suite.run(validation_data=sample_frame)

    rendered = render_html_report(result)

    assert 'data-view-target="model-card"' in rendered
    assert 'data-view="model-card" hidden' in rendered
    assert 'data-view-target="methodology"' not in rendered


def test_default_report_title_is_localized_and_inert_actions_are_absent(
    sample_frame: pl.DataFrame,
) -> None:
    suite = _suite()
    suite.config.report.language = "es"
    suite.config.report.include_charts = False

    rendered = render_html_report(suite.run(validation_data=sample_frame))

    assert "<title>Reporte de validacion PD</title>" in rendered
    assert "<h1>Reporte de validacion PD</h1>" in rendered
    assert 'class="header-actions"' not in rendered
    assert "Share Report" not in rendered


def test_html_displays_all_calibration_metrics(sample_frame: pl.DataFrame) -> None:
    suite = _suite()
    suite.config.report.include_charts = False

    rendered = render_html_report(suite.run(validation_data=sample_frame))

    for label in [
        "MCE",
        "Calibration-in-the-large",
        "Calibration Intercept",
        "Calibration Slope",
    ]:
        assert f'<span class="kpi-label">{label}</span>' in rendered


def test_psi_threshold_legend_uses_metric_thresholds(sample_frame: pl.DataFrame) -> None:
    result = _suite(segments=["segment"]).run_drift(
        reference_data=sample_frame, current_data=sample_frame
    )
    result.metrics["psi_pd"] = replace(
        result.metrics["psi_pd"], threshold_warning=0.07, threshold_critical=0.18
    )

    rendered = _threshold_legend(result, translations("en"))

    assert "Warning Threshold 0.07" in rendered
    assert "Critical Threshold 0.18" in rendered
    assert "0.10" not in rendered
    assert "0.25" not in rendered


def test_segment_drift_is_grouped_without_cross_variable_total(
    sample_frame: pl.DataFrame,
) -> None:
    result = _suite(segments=["segment"]).run_drift(
        reference_data=sample_frame, current_data=sample_frame
    )
    result.tables["segment_drift"] = pl.DataFrame(
        [
            _segment_row("region", "north", 60, 55, 0.60, 0.55, 0.01),
            _segment_row("region", "south", 40, 45, 0.40, 0.45, 0.02),
            _segment_row("product", "a", 70, 65, 0.70, 0.65, 0.04),
            _segment_row("product", "b", 30, 35, 0.30, 0.35, 0.05),
        ]
    )

    rendered = _segment_drift_panel(result, translations("en"))

    assert "Segment Drift: region" in rendered
    assert "Segment Drift: product" in rendered
    assert rendered.count(">100.00%</td>") == 4
    assert ">200.00%</td>" not in rendered
    assert ">0.03</td>" in rendered
    assert ">0.09</td>" in rendered


def _segment_row(
    column: str,
    segment: str,
    reference_count: int,
    current_count: int,
    reference_share: float,
    current_share: float,
    population_psi: float,
) -> dict[str, object]:
    return {
        "segment_column": column,
        "segment": segment,
        "reference_count": reference_count,
        "current_count": current_count,
        "reference_share": reference_share,
        "current_share": current_share,
        "population_psi": population_psi,
        "reference_bad_rate": 0.10,
        "current_bad_rate": 0.12,
        "reference_avg_pd": 0.11,
        "current_avg_pd": 0.13,
        "bad_rate_delta": 0.02,
        "avg_pd_delta": 0.02,
    }


def test_drift_kpi_uses_current_value_as_primary() -> None:
    metric = MetricResult(
        "auc",
        0.82,
        Status.WARNING,
        reference_value=0.82,
        current_value=0.61,
        delta=-0.21,
    )

    rendered = _kpi_card(metric, "AUC", prefer_current=True)

    assert 'data-value-source="current">0.61</strong>' in rendered
    assert "<b>Ref</b>0.82" in rendered
    assert "<b>Cur</b>0.61" in rendered


def test_charts_compare_current_tables_when_available(sample_frame: pl.DataFrame) -> None:
    result = _suite().run(validation_data=sample_frame)
    result.metadata["analysis_type"] = "drift"
    result.tables["current_lift_table"] = result.tables["lift_table"].with_columns(
        (pl.col("bad_rate") * 0.5).alias("bad_rate"),
        (pl.col("cumulative_event_capture") * 0.8).alias("cumulative_event_capture"),
    )
    result.tables["current_calibration_bins"] = result.tables["calibration_bins"].with_columns(
        (pl.col("observed_rate") * 0.5).alias("observed_rate")
    )
    locale = translations("en")

    lift = _lift_chart(result, locale)
    bad_rate = _bad_rate_chart(result, locale)
    calibration = _calibration_chart(result, locale)

    assert lift is not None
    assert bad_rate is not None
    assert calibration is not None
    assert [trace["name"] for trace in lift.to_dict()["data"]] == [
        "Reference event capture",
        "Reference population share",
        "Current event capture",
        "Current population share",
    ]
    assert [trace["name"] for trace in bad_rate.to_dict()["data"]] == [
        "Reference bad rate",
        "Current bad rate",
    ]
    assert [trace["name"] for trace in calibration.to_dict()["data"]] == [
        "Reference observed default rate",
        "Current observed default rate",
        "Perfect calibration",
    ]
    assert calibration.layout.yaxis.scaleanchor == "x"
    assert calibration.layout.xaxis.range == calibration.layout.yaxis.range
