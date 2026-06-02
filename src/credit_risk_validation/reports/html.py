"""Self-contained HTML reporting."""

from html import escape
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.io import to_html

from credit_risk_validation.reports.i18n import t, translations

if TYPE_CHECKING:
    from credit_risk_validation.results import PDValidationResult


def render_html_report(result: "PDValidationResult") -> str:
    """Renderiza un reporte HTML autocontenido con tablas agregadas."""

    locale = translations(result.config.report.language)
    title = escape(result.config.report.title)
    model = result.config.model
    columns = result.config.columns
    metrics_rows = "\n".join(
        "<tr>"
        f"<td>{escape(metric.name)}</td>"
        f"<td>{_format_metric_value(metric.value)}</td>"
        f"<td>{_format_metric_value(metric.reference_value)}</td>"
        f"<td>{_format_metric_value(metric.current_value)}</td>"
        f"<td>{_format_metric_value(metric.delta)}</td>"
        f"<td>{_format_metric_value(metric.threshold_warning)}</td>"
        f"<td>{_format_metric_value(metric.threshold_critical)}</td>"
        f"<td>{escape(metric.status.value)}</td>"
        f"<td>{escape(metric.message)}</td>"
        "</tr>"
        for metric in result.metrics.values()
    )
    checks_rows = "\n".join(
        "<tr>"
        f"<td>{escape(check.name)}</td>"
        f"<td>{escape(check.status.value)}</td>"
        f"<td>{escape(check.message)}</td>"
        "</tr>"
        for check in result.checks
    )
    table_sections = "\n".join(
        _table_section(name, table.to_dicts(), locale) for name, table in result.tables.items()
    )
    chart_sections = (
        _chart_sections(result, locale)
        if result.config.report.include_charts
        else {"discrimination": "", "calibration": "", "stability": "", "segments": ""}
    )
    model_card_text = escape(
        t(locale, "model_card_body").format(
            purpose=str(model.purpose or t(locale, "not_specified")),
            owner=str(model.owner or t(locale, "not_specified")),
        )
    )
    model_card_section = (
        f"""
    <section>
      <h2>{t(locale, "model_card")}</h2>
      <p>{model_card_text}</p>
    </section>"""
        if result.config.report.include_model_card
        else ""
    )
    methodology_section = (
        f"""
    <section>
      <h2>{t(locale, "methodology_title")}</h2>
      <p>{t(locale, "methodology_body")}</p>
    </section>"""
        if result.config.report.include_methodology
        else ""
    )
    return f"""<!doctype html>
<html lang="{escape(result.config.report.language)}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; color: #1f2937; background: #f8fafc; }}
    header {{ background: #0f172a; color: white; padding: 32px 40px; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    section {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; margin: 18px 0; padding: 20px; }}
    h1, h2 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f1f5f9; }}
    .status {{ font-weight: 700; }}
    .disclaimer {{ color: #475569; font-size: 13px; }}
    .chart {{ margin-top: 16px; }}
    code {{ background: #f1f5f9; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p>{t(locale, "slogan")}</p>
    <p class="status">{t(locale, "overall_status")}: {escape(result.status.value)}</p>
  </header>
  <main>
    <section>
      <h2>{t(locale, "cover")}</h2>
      <table>
        <tbody>
          <tr><th>{t(locale, "model")}</th><td>{escape(model.name)}</td></tr>
          <tr><th>{t(locale, "model_version")}</th><td>{escape(model.version)}</td></tr>
          <tr><th>{t(locale, "library_version")}</th><td>{_metadata_value(result, "library_version", locale)}</td></tr>
          <tr><th>{t(locale, "generated_at")}</th><td>{escape(result.created_at)}</td></tr>
          <tr><th>{t(locale, "pd_horizon")}</th><td>{escape(model.horizon)}</td></tr>
          <tr><th>{t(locale, "target_column")}</th><td><code>{escape(columns.target)}</code></td></tr>
          <tr><th>{t(locale, "pd_column")}</th><td><code>{escape(columns.pd)}</code></td></tr>
          <tr><th>{t(locale, "score_column")}</th><td><code>{escape(str(columns.score))}</code></td></tr>
          <tr><th>{t(locale, "reference_rows")}</th><td>{_metadata_value(result, "reference_rows", locale)}</td></tr>
          <tr><th>{t(locale, "current_rows")}</th><td>{_metadata_value(result, "current_rows", locale)}</td></tr>
          <tr><th>{t(locale, "config_hash")}</th><td><code>{_metadata_value(result, "config_sha256", locale)}</code></td></tr>
          <tr><th>{t(locale, "reference_schema_hash")}</th><td><code>{_metadata_value(result, "reference_schema_sha256", locale)}</code></td></tr>
          <tr><th>{t(locale, "current_schema_hash")}</th><td><code>{_metadata_value(result, "current_schema_sha256", locale)}</code></td></tr>
        </tbody>
      </table>
    </section>
    <section>
      <h2>{t(locale, "executive_summary")}</h2>
      <p>{t(locale, "executive_summary_body")}</p>
      <p class="disclaimer">{t(locale, "disclaimer")}</p>
    </section>
    <section>
      <h2>{t(locale, "data_quality")}</h2>
      <table><thead><tr><th>{t(locale, "check")}</th><th>{t(locale, "status")}</th><th>{t(locale, "message")}</th></tr></thead><tbody>{checks_rows}</tbody></table>
    </section>
    <section>
      <h2>{t(locale, "metrics")}</h2>
      <table><thead><tr><th>{t(locale, "metric")}</th><th>{t(locale, "value")}</th><th>{t(locale, "reference_value")}</th><th>{t(locale, "current_value")}</th><th>{t(locale, "delta")}</th><th>{t(locale, "threshold_warning")}</th><th>{t(locale, "threshold_critical")}</th><th>{t(locale, "status")}</th><th>{t(locale, "message")}</th></tr></thead><tbody>{metrics_rows}</tbody></table>
    </section>
    <section>
      <h2>{t(locale, "discrimination_title")}</h2>
      <p>{t(locale, "discrimination_body")}</p>
      {chart_sections["discrimination"]}
    </section>
    <section>
      <h2>{t(locale, "calibration_title")}</h2>
      <p>{t(locale, "calibration_body")}</p>
      {chart_sections["calibration"]}
    </section>
    <section>
      <h2>{t(locale, "stability_title")}</h2>
      <p>{t(locale, "stability_body")}</p>
      {chart_sections["stability"]}
    </section>
    <section>
      <h2>{t(locale, "segment_analysis_title")}</h2>
      <p>{t(locale, "segment_analysis_body")}</p>
      {chart_sections["segments"]}
    </section>
    <section>
      <h2>{t(locale, "champion_title")}</h2>
      <p>{t(locale, "champion_body")}</p>
    </section>
    {model_card_section}
    {table_sections}
    {methodology_section}
    <section>
      <h2>{t(locale, "raw_outputs")}</h2>
      <p>{t(locale, "raw_outputs_body")}</p>
    </section>
  </main>
</body>
</html>
"""


def _chart_sections(result: "PDValidationResult", locale: dict[str, str]) -> dict[str, str]:
    include_plotlyjs = True
    sections: dict[str, str] = {
        "discrimination": "",
        "calibration": "",
        "stability": "",
        "segments": "",
    }
    charts = [
        ("discrimination", _lift_chart(result, locale)),
        ("discrimination", _bad_rate_chart(result, locale)),
        ("calibration", _calibration_chart(result, locale)),
        (
            "stability",
            _psi_chart(result, "psi_pd", t(locale, "pd_psi_by_bin"), locale),
        ),
        (
            "stability",
            _psi_chart(result, "psi_score", t(locale, "score_psi_by_bin"), locale),
        ),
        ("segments", _segment_chart(result, locale)),
    ]
    for section, figure in charts:
        if figure is None:
            continue
        sections[section] += _figure_html(figure, include_plotlyjs=include_plotlyjs)
        include_plotlyjs = False
    return sections


def _metadata_value(result: "PDValidationResult", key: str, locale: dict[str, str]) -> str:
    value = result.metadata.get(key)
    return escape(t(locale, "not_available") if value is None else str(value))


def _format_metric_value(value: float | None) -> str:
    return "" if value is None else escape(f"{value:.6g}")


def _lift_chart(result: "PDValidationResult", locale: dict[str, str]) -> go.Figure | None:
    table = result.tables.get("lift_table")
    if table is None or table.is_empty():
        return None
    rows = table.to_dicts()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[row["bin"] for row in rows],
            y=[row["cumulative_event_capture"] for row in rows],
            mode="lines+markers",
            name="Cumulative event capture",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[row["bin"] for row in rows],
            y=[row["cumulative_population_share"] for row in rows],
            mode="lines+markers",
            name="Cumulative population share",
        )
    )
    figure.update_layout(
        title=t(locale, "lift_capture_title"),
        xaxis_title=t(locale, "risk_bin"),
        yaxis_title=t(locale, "share"),
        yaxis_tickformat=".0%",
        height=360,
    )
    return figure


def _bad_rate_chart(result: "PDValidationResult", locale: dict[str, str]) -> go.Figure | None:
    table = result.tables.get("lift_table")
    if table is None or table.is_empty():
        return None
    rows = table.to_dicts()
    figure = go.Figure(
        go.Bar(
            x=[row["bin"] for row in rows],
            y=[row["bad_rate"] for row in rows],
            name="Bad rate",
        )
    )
    figure.update_layout(
        title=t(locale, "bad_rate_by_risk_bin"),
        xaxis_title=t(locale, "risk_bin"),
        yaxis_title="Bad rate",
        yaxis_tickformat=".0%",
        height=360,
    )
    return figure


def _calibration_chart(result: "PDValidationResult", locale: dict[str, str]) -> go.Figure | None:
    table = result.tables.get("calibration_bins")
    if table is None or table.is_empty():
        return None
    rows = table.to_dicts()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[row["mean_pd"] for row in rows],
            y=[row["observed_rate"] for row in rows],
            mode="markers+lines",
            name="Observed default rate",
        )
    )
    max_axis = max(
        [float(row["mean_pd"] or 0) for row in rows]
        + [float(row["observed_rate"] or 0) for row in rows]
    )
    figure.add_trace(
        go.Scatter(
            x=[0, max_axis],
            y=[0, max_axis],
            mode="lines",
            name="Perfect calibration",
            line={"dash": "dash"},
        )
    )
    figure.update_layout(
        title=t(locale, "calibration_chart_title"),
        xaxis_title=t(locale, "calibration_x"),
        yaxis_title=t(locale, "calibration_y"),
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        height=380,
    )
    return figure


def _psi_chart(
    result: "PDValidationResult",
    table_name: str,
    title: str,
    locale: dict[str, str],
) -> go.Figure | None:
    table = result.tables.get(table_name)
    if table is None or table.is_empty():
        return None
    rows = table.to_dicts()
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[row["bin"] for row in rows],
            y=[row["reference_share"] for row in rows],
            name="Reference share",
        )
    )
    figure.add_trace(
        go.Bar(
            x=[row["bin"] for row in rows],
            y=[row["current_share"] for row in rows],
            name="Current share",
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Bin",
        yaxis_title=t(locale, "population_share"),
        yaxis_tickformat=".0%",
        barmode="group",
        height=360,
    )
    return figure


def _segment_chart(result: "PDValidationResult", locale: dict[str, str]) -> go.Figure | None:
    table = result.tables.get("segment_metrics")
    if table is None or table.is_empty() or "segment" not in table.columns:
        return None
    rows = table.to_dicts()
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[str(row["segment"]) for row in rows],
            y=[row.get("observed_rate") for row in rows],
            name="Observed rate",
        )
    )
    figure.add_trace(
        go.Bar(
            x=[str(row["segment"]) for row in rows],
            y=[row.get("mean_pd") for row in rows],
            name="Mean PD",
        )
    )
    figure.update_layout(
        title=t(locale, "segment_calibration_summary"),
        xaxis_title=t(locale, "segment_x"),
        yaxis_title="Rate",
        yaxis_tickformat=".0%",
        barmode="group",
        height=380,
    )
    return figure


def _figure_html(figure: go.Figure, *, include_plotlyjs: bool) -> str:
    return (
        '<div class="chart">'
        + to_html(
            figure,
            include_plotlyjs=include_plotlyjs,
            full_html=False,
            config={"displayModeBar": False, "responsive": True},
        )
        + "</div>"
    )


def _table_section(name: str, rows: list[dict[str, object]], locale: dict[str, str]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    header_html = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body_html = "\n".join(
        "<tr>"
        + "".join(f"<td>{escape(_format_cell(row.get(header)))}</td>" for header in headers)
        + "</tr>"
        for row in rows[:200]
    )
    suffix = f"<p>{t(locale, 'table_truncated')}</p>" if len(rows) > 200 else ""
    return (
        f"<section><h2>{escape(name.replace('_', ' ').title())}</h2>"
        f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>{suffix}</section>"
    )


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
