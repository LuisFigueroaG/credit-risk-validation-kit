"""Self-contained HTML reporting."""

from html import escape
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.io import to_html

if TYPE_CHECKING:
    from credit_risk_validation.results import PDValidationResult


def render_html_report(result: "PDValidationResult") -> str:
    """Renderiza un reporte HTML autocontenido con tablas agregadas."""

    title = escape(result.config.report.title)
    model = result.config.model
    columns = result.config.columns
    metrics_rows = "\n".join(
        "<tr>"
        f"<td>{escape(metric.name)}</td>"
        f"<td>{'' if metric.value is None else escape(f'{metric.value:.6g}')}</td>"
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
        _table_section(name, table.to_dicts()) for name, table in result.tables.items()
    )
    chart_sections = _chart_sections(result)
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
    <p>The missing validation and reporting layer for credit risk models.</p>
    <p class="status">Overall status: {escape(result.status.value)}</p>
  </header>
  <main>
    <section>
      <h2>Cover</h2>
      <table>
        <tbody>
          <tr><th>Model</th><td>{escape(model.name)}</td></tr>
          <tr><th>Model Version</th><td>{escape(model.version)}</td></tr>
          <tr><th>Library Version</th><td>{_metadata_value(result, "library_version")}</td></tr>
          <tr><th>Generated At</th><td>{escape(result.created_at)}</td></tr>
          <tr><th>PD Horizon</th><td>{escape(model.horizon)}</td></tr>
          <tr><th>Target Column</th><td><code>{escape(columns.target)}</code></td></tr>
          <tr><th>PD Column</th><td><code>{escape(columns.pd)}</code></td></tr>
          <tr><th>Score Column</th><td><code>{escape(str(columns.score))}</code></td></tr>
          <tr><th>Reference Rows</th><td>{_metadata_value(result, "reference_rows")}</td></tr>
          <tr><th>Current Rows</th><td>{_metadata_value(result, "current_rows")}</td></tr>
          <tr><th>Config SHA-256</th><td><code>{_metadata_value(result, "config_sha256")}</code></td></tr>
          <tr><th>Reference Schema SHA-256</th><td><code>{_metadata_value(result, "reference_schema_sha256")}</code></td></tr>
          <tr><th>Current Schema SHA-256</th><td><code>{_metadata_value(result, "current_schema_sha256")}</code></td></tr>
        </tbody>
      </table>
    </section>
    <section>
      <h2>Executive Summary</h2>
      <p>This report supports validation, generates quantitative evidence, documents findings,
      and monitors stability and calibration for binary PD models.</p>
      <p class="disclaimer">It does not certify regulatory compliance, approve models, calculate
      regulatory capital, or calculate official provisions.</p>
    </section>
    <section>
      <h2>Data Quality</h2>
      <table><thead><tr><th>Check</th><th>Status</th><th>Message</th></tr></thead><tbody>{checks_rows}</tbody></table>
    </section>
    <section>
      <h2>Metrics</h2>
      <table><thead><tr><th>Metric</th><th>Value</th><th>Status</th><th>Message</th></tr></thead><tbody>{metrics_rows}</tbody></table>
    </section>
    <section>
      <h2>Discrimination</h2>
      <p>AUC, Gini, KS, lift, bad-rate and capture-rate outputs are reported as aggregate metrics and tables.</p>
      {chart_sections["discrimination"]}
    </section>
    <section>
      <h2>Calibration</h2>
      <p>Brier Score, Log Loss, ECE, MCE, O/E ratio and calibration bins are reported when calculable.</p>
      {chart_sections["calibration"]}
    </section>
    <section>
      <h2>Stability</h2>
      <p>PSI for PD and score is reported when reference and current samples are provided.</p>
      {chart_sections["stability"]}
    </section>
    <section>
      <h2>Segment Analysis</h2>
      <p>Configured segment metrics are reported only with sufficient events and non-events.</p>
      {chart_sections["segments"]}
    </section>
    <section>
      <h2>Champion vs Challenger</h2>
      <p>Side-by-side model comparison is experimental and never recommends automatic replacement.</p>
    </section>
    <section>
      <h2>Model Card</h2>
      <p>Purpose: {escape(str(model.purpose or "not specified"))}. Owner: {escape(str(model.owner or "not specified"))}.</p>
    </section>
    {table_sections}
    <section>
      <h2>Methodology Appendix</h2>
      <p>AUC, Gini and KS summarize discrimination. Brier, Log Loss, calibration bins,
      ECE and O/E summarize calibration. PSI summarizes distribution stability between
      reference and current samples.</p>
    </section>
    <section>
      <h2>Raw Outputs</h2>
      <p>JSON and CSV/Parquet outputs are written separately by the configured exporters. HTML includes aggregate output previews only.</p>
    </section>
  </main>
</body>
</html>
"""


def _chart_sections(result: "PDValidationResult") -> dict[str, str]:
    include_plotlyjs = True
    sections: dict[str, str] = {
        "discrimination": "",
        "calibration": "",
        "stability": "",
        "segments": "",
    }
    charts = [
        ("discrimination", _lift_chart(result)),
        ("discrimination", _bad_rate_chart(result)),
        ("calibration", _calibration_chart(result)),
        ("stability", _psi_chart(result, "psi_pd", "PD PSI by Bin")),
        ("stability", _psi_chart(result, "psi_score", "Score PSI by Bin")),
        ("segments", _segment_chart(result)),
    ]
    for section, figure in charts:
        if figure is None:
            continue
        sections[section] += _figure_html(figure, include_plotlyjs=include_plotlyjs)
        include_plotlyjs = False
    return sections


def _metadata_value(result: "PDValidationResult", key: str) -> str:
    value = result.metadata.get(key)
    return escape("not available" if value is None else str(value))


def _lift_chart(result: "PDValidationResult") -> go.Figure | None:
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
        title="Lift and Event Capture",
        xaxis_title="Risk bin",
        yaxis_title="Share",
        yaxis_tickformat=".0%",
        height=360,
    )
    return figure


def _bad_rate_chart(result: "PDValidationResult") -> go.Figure | None:
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
        title="Bad Rate by Risk Bin",
        xaxis_title="Risk bin",
        yaxis_title="Bad rate",
        yaxis_tickformat=".0%",
        height=360,
    )
    return figure


def _calibration_chart(result: "PDValidationResult") -> go.Figure | None:
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
        title="Calibration: Average PD vs Observed Default Rate",
        xaxis_title="Average predicted PD",
        yaxis_title="Observed default rate",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        height=380,
    )
    return figure


def _psi_chart(result: "PDValidationResult", table_name: str, title: str) -> go.Figure | None:
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
        yaxis_title="Population share",
        yaxis_tickformat=".0%",
        barmode="group",
        height=360,
    )
    return figure


def _segment_chart(result: "PDValidationResult") -> go.Figure | None:
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
        title="Segment Calibration Summary",
        xaxis_title="Segment",
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


def _table_section(name: str, rows: list[dict[str, object]]) -> str:
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
    suffix = "<p>Table truncated to first 200 aggregate rows.</p>" if len(rows) > 200 else ""
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
