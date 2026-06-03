"""Self-contained HTML reporting."""

from html import escape
from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl
from plotly.io import to_html

from credit_risk_validation.reports.i18n import t, translations

if TYPE_CHECKING:
    from credit_risk_validation.results import PDValidationResult
    from credit_risk_validation.schemas import MetricResult


KEY_METRICS = [
    ("auc", "AUC"),
    ("gini", "Gini"),
    ("ks", "KS"),
    ("brier", "Brier"),
    ("log_loss", "Log Loss"),
    ("ece", "ECE"),
    ("oe_ratio", "O/E Ratio"),
    ("psi_pd", "PSI PD"),
    ("psi_score", "PSI Score"),
]


def render_html_report(result: "PDValidationResult") -> str:
    """Renderiza un reporte HTML autocontenido con tablas agregadas."""

    locale = translations(result.config.report.language)
    title = escape(result.config.report.title)
    model = result.config.model
    columns = result.config.columns
    chart_sections = (
        _chart_sections(result, locale)
        if result.config.report.include_charts
        else {"discrimination": "", "calibration": "", "stability": "", "segments": ""}
    )
    model_card_section = _model_card_section(result, locale)
    methodology_section = _methodology_section(result, locale)
    table_sections = "\n".join(
        _table_section(name, table.to_dicts(), locale) for name, table in result.tables.items()
    )
    default_view = _default_view(result)

    return f"""<!doctype html>
<html lang="{escape(result.config.report.language)}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{_styles()}</style>
</head>
<body>
  <div class="app-shell">
    {_sidebar(default_view, include_methodology=result.config.report.include_methodology)}
    <div class="report-window">
      <header class="topbar">
        <div>
          <h1>{title}</h1>
          <div class="meta-line">
            <span>{t(locale, "model")}: {escape(model.name)}</span>
            <span>{t(locale, "model_version")}: {escape(model.version)}</span>
            <span>{t(locale, "generated_at")}: {escape(_short_datetime(result.created_at))}</span>
          </div>
        </div>
        <div class="header-actions" aria-label="Report actions">
          <button type="button">{_icon("download")} Download</button>
          <button type="button" class="primary">{_icon("share")} Share Report</button>
        </div>
      </header>
      {_tabs(locale, default_view, include_methodology=result.config.report.include_methodology)}
      <main>
        <section id="view-overview" {_view_attrs("overview", default_view)}>
          <div class="section-heading">
            <div>
              <p class="eyebrow">{t(locale, "executive_summary")}</p>
              <h2>{t(locale, "overall_status")}</h2>
            </div>
            <span class="status-pill {escape(result.status.value.lower())}">{escape(result.status.value)}</span>
          </div>
          {_status_banner(result.status.value, locale)}
          {_kpi_grid(result, KEY_METRICS, class_name="overview-kpi-grid")}
          <div class="overview-grid">
            <article class="card findings-card">
              <div class="card-title">
                <h3>{t(locale, "executive_summary")}</h3>
                <span>{escape(result.metadata.get("privacy", ""))}</span>
              </div>
              <p>{t(locale, "executive_summary_body")}</p>
              {_findings_list(result)}
              <p class="disclaimer">{t(locale, "disclaimer")}</p>
            </article>
            <article class="card compact-card">
              <div class="card-title">
                <h3>{t(locale, "cover")}</h3>
              </div>
              <dl class="cover-list">
                <div><dt>{t(locale, "library_version")}</dt><dd>{_metadata_value(result, "library_version", locale)}</dd></div>
                <div><dt>{t(locale, "pd_horizon")}</dt><dd>{escape(model.horizon)}</dd></div>
                <div><dt>{t(locale, "target_column")}</dt><dd><code>{escape(columns.target)}</code></dd></div>
                <div><dt>{t(locale, "pd_column")}</dt><dd><code>{escape(columns.pd)}</code></dd></div>
                <div><dt>{t(locale, "score_column")}</dt><dd><code>{escape(str(columns.score))}</code></dd></div>
                <div><dt>{t(locale, "reference_rows")}</dt><dd>{_metadata_value(result, "reference_rows", locale)}</dd></div>
                <div><dt>{t(locale, "current_rows")}</dt><dd>{_metadata_value(result, "current_rows", locale)}</dd></div>
                <div><dt>{t(locale, "config_hash")}</dt><dd><code>{_metadata_value(result, "config_sha256", locale)}</code></dd></div>
                <div><dt>{t(locale, "reference_schema_hash")}</dt><dd><code>{_metadata_value(result, "reference_schema_sha256", locale)}</code></dd></div>
                <div><dt>{t(locale, "current_schema_hash")}</dt><dd><code>{_metadata_value(result, "current_schema_sha256", locale)}</code></dd></div>
              </dl>
            </article>
          </div>
        </section>

        <section id="view-discrimination" {_view_attrs("discrimination", default_view)}>
          <div class="section-heading">
            <div>
              <p class="eyebrow">Model ranking</p>
              <h2>{t(locale, "discrimination_title")}</h2>
              <p>{t(locale, "discrimination_body")}</p>
            </div>
          </div>
          {_kpi_grid(result, [("auc", "AUC"), ("gini", "Gini"), ("ks", "KS")])}
          <div class="chart-grid two-col">{chart_sections["discrimination"]}</div>
          {_table_panel("lift_table", result.tables.get("lift_table"), locale)}
        </section>

        <section id="view-calibration" {_view_attrs("calibration", default_view)}>
          <div class="section-heading">
            <div>
              <p class="eyebrow">Probability quality</p>
              <h2>{t(locale, "calibration_title")}</h2>
              <p>{t(locale, "calibration_body")}</p>
            </div>
          </div>
          {_kpi_grid(result, [("brier", "Brier"), ("log_loss", "Log Loss"), ("ece", "ECE"), ("oe_ratio", "O/E Ratio")])}
          <div class="panel-grid calibration-grid">
            <article class="card chart-card">{chart_sections["calibration"]}</article>
            <article class="card insight-card">
              <div class="card-title"><h3>Calibration Evidence</h3></div>
              {_metric_delta_list(result, ["brier", "log_loss", "ece", "oe_ratio"])}
            </article>
          </div>
          {_table_panel("calibration_bins", result.tables.get("calibration_bins"), locale)}
        </section>

        <section id="view-stability" {_view_attrs("stability", default_view)}>
          <div class="section-heading">
            <div>
              <p class="eyebrow">Population drift</p>
              <h2>{t(locale, "stability_title")}</h2>
              <p>{t(locale, "stability_body")}</p>
            </div>
          </div>
          {_stability_banner(result)}
          <div class="panel-grid stability-grid">
            {_psi_variable_card(result, locale)}
            {_distribution_card(result, "psi_pd", "PD Distribution", "PD")}
            {_distribution_card(result, "psi_score", "Score Distribution", "Score")}
          </div>
          <span class="sr-only">{t(locale, "pd_psi_by_bin")} {t(locale, "score_psi_by_bin")}</span>
          {_segment_drift_panel(result, locale)}
        </section>

        <section id="view-segments" {_view_attrs("segments", default_view)}>
          <div class="section-heading">
            <div>
              <p class="eyebrow">Population slices</p>
              <h2>{t(locale, "segment_analysis_title")}</h2>
              <p>{t(locale, "segment_analysis_body")}</p>
            </div>
          </div>
          <div class="panel-grid segments-grid">
            <article class="card chart-card">{chart_sections["segments"]}</article>
            {_segment_summary_card(result)}
          </div>
          {_table_panel("segment_metrics", result.tables.get("segment_metrics"), locale)}
          {_table_panel("current_segment_analysis", result.tables.get("current_segment_analysis"), locale)}
        </section>

        <section id="view-data-quality" {_view_attrs("data-quality", default_view)}>
          <div class="section-heading">
            <div>
              <p class="eyebrow">Controls</p>
              <h2>{t(locale, "data_quality")}</h2>
            </div>
          </div>
          <div class="panel-grid data-grid">
            {_checks_panel(result, locale)}
            {_table_panel("data_quality", result.tables.get("data_quality"), locale, compact=True)}
          </div>
        </section>

        {methodology_section}
        {model_card_section}

        <section id="view-tables" {_view_attrs("tables", default_view)}>
          <div class="section-heading">
            <div>
              <p class="eyebrow">{t(locale, "raw_outputs")}</p>
              <h2>Aggregate Tables</h2>
              <p>{t(locale, "raw_outputs_body")}</p>
            </div>
          </div>
          {table_sections}
        </section>
      </main>
    </div>
  </div>
  <script>{_scripts(default_view)}</script>
</body>
</html>
"""


def _styles() -> str:
    return """
    :root {
      --navy-950: #061936;
      --navy-900: #08244a;
      --navy-800: #0f2f5f;
      --blue-700: #1d4ed8;
      --blue-600: #2563eb;
      --blue-500: #3b82f6;
      --blue-100: #dbeafe;
      --blue-50: #eff6ff;
      --slate-950: #0f172a;
      --slate-700: #334155;
      --slate-600: #475569;
      --slate-500: #64748b;
      --slate-300: #cbd5e1;
      --slate-200: #e2e8f0;
      --slate-100: #f1f5f9;
      --slate-50: #f8fafc;
      --white: #ffffff;
      --critical: #dc2626;
      --critical-soft: #fff1f2;
      --warning: #f59e0b;
      --warning-soft: #fffbeb;
      --ok: #16a34a;
      --ok-soft: #f0fdf4;
      --radius: 8px;
      --shadow: 0 18px 50px rgba(15, 23, 42, 0.09);
      --soft-shadow: 0 8px 26px rgba(15, 23, 42, 0.06);
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-width: 320px;
      color: var(--slate-950);
      background: #dfe7f1;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }

    a { color: inherit; text-decoration: none; }
    button { font: inherit; }
    code {
      display: inline-block;
      max-width: 100%;
      padding: 2px 6px;
      overflow-wrap: anywhere;
      color: var(--navy-900);
      background: var(--blue-50);
      border: 1px solid var(--blue-100);
      border-radius: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
    }

    .app-shell {
      display: grid;
      grid-template-columns: 96px minmax(0, 1fr);
      min-height: 100vh;
      padding: 0;
      gap: 0;
    }

    .sidebar {
      position: sticky;
      top: 0;
      height: 100vh;
      padding: 26px 20px;
      background: linear-gradient(180deg, var(--navy-950), #042a52);
      box-shadow: 10px 0 34px rgba(2, 8, 23, 0.22);
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 18px;
      color: white;
    }

    .brand-mark,
    .side-link {
      display: grid;
      place-items: center;
      width: 48px;
      height: 48px;
      border-radius: 8px;
    }

    .brand-mark {
      background: linear-gradient(145deg, #0ea5e9, var(--blue-700));
      box-shadow: 0 10px 24px rgba(37, 99, 235, 0.42);
    }

    .side-nav {
      display: flex;
      flex: 1;
      flex-direction: column;
      gap: 10px;
      align-items: center;
    }

    .side-link {
      color: rgba(255, 255, 255, 0.72);
      transition: background 150ms ease, color 150ms ease, transform 150ms ease;
    }

    .side-link:hover,
    .side-link.active {
      background: rgba(37, 99, 235, 0.64);
      color: white;
      transform: translateY(-1px);
    }

    .sidebar svg,
    .tab svg,
    button svg {
      width: 20px;
      height: 20px;
      stroke: currentColor;
      stroke-width: 1.9;
      fill: none;
      stroke-linecap: round;
      stroke-linejoin: round;
      flex: 0 0 auto;
    }

    .brand-mark svg { fill: rgba(255, 255, 255, 0.95); stroke: none; }

    .report-window {
      width: auto;
      min-height: calc(100vh - 32px);
      margin: 16px 18px 16px 10px;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(203, 213, 225, 0.92);
      border-radius: 10px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 26px 34px 20px;
      border-bottom: 1px solid var(--slate-200);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(255, 255, 255, 0.88));
    }

    h1, h2, h3, p { margin-top: 0; }
    h1 {
      margin-bottom: 8px;
      font-size: clamp(24px, 2vw, 34px);
      line-height: 1.1;
      font-weight: 780;
      letter-spacing: 0;
      color: var(--slate-950);
    }

    h2 {
      margin-bottom: 8px;
      font-size: clamp(22px, 1.6vw, 30px);
      line-height: 1.15;
      letter-spacing: 0;
    }

    h3 {
      margin-bottom: 0;
      font-size: 16px;
      line-height: 1.25;
      letter-spacing: 0;
    }

    p {
      color: var(--slate-600);
      line-height: 1.55;
    }

    .meta-line {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      color: var(--slate-600);
      font-size: 14px;
    }

    .meta-line span + span::before {
      content: "";
      display: inline-block;
      width: 4px;
      height: 4px;
      margin-right: 18px;
      vertical-align: middle;
      background: var(--slate-300);
      border-radius: 999px;
    }

    .header-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      justify-content: flex-end;
    }

    .header-actions button {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      min-height: 44px;
      padding: 0 16px;
      color: var(--slate-950);
      background: white;
      border: 1px solid var(--slate-200);
      border-radius: 8px;
      box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
    }

    .header-actions .primary {
      color: white;
      background: linear-gradient(180deg, #3b82f6, #2563eb);
      border-color: #2563eb;
    }

    .tabs {
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding: 0 36px;
      background: rgba(255, 255, 255, 0.95);
      border-bottom: 1px solid var(--slate-200);
      backdrop-filter: blur(10px);
    }

    .tab {
      position: relative;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      min-height: 70px;
      padding: 0 18px;
      color: var(--slate-700);
      white-space: nowrap;
    }

    .tab:hover,
    .tab.active {
      color: var(--blue-700);
    }

    .tab.active::after {
      position: absolute;
      right: 14px;
      bottom: 0;
      left: 14px;
      height: 3px;
      content: "";
      background: var(--blue-600);
      border-radius: 999px 999px 0 0;
    }

    main {
      padding: 22px 34px 24px;
    }

    .view-section {
      padding: 0 0 8px;
    }

    .view-section[hidden] {
      display: none !important;
    }
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }

    .section-heading {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      margin: 0 0 16px;
    }

    .section-heading p { margin-bottom: 0; }
    .eyebrow {
      margin-bottom: 7px;
      color: var(--blue-700);
      font-size: 12px;
      font-weight: 760;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 0 11px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 760;
      border: 1px solid transparent;
    }

    .status-pill.ok { color: #166534; background: var(--ok-soft); border-color: #bbf7d0; }
    .status-pill.warning { color: #92400e; background: var(--warning-soft); border-color: #fde68a; }
    .status-pill.critical,
    .status-pill.error { color: #991b1b; background: var(--critical-soft); border-color: #fecaca; }
    .status-pill.insufficient_data,
    .status-pill.not_applicable { color: var(--slate-600); background: var(--slate-100); border-color: var(--slate-200); }

    .status-banner {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 18px;
      align-items: center;
      min-height: 88px;
      padding: 18px 24px;
      border: 1px solid var(--blue-100);
      border-radius: var(--radius);
      background: linear-gradient(90deg, var(--blue-50), rgba(255, 255, 255, 0.94));
      box-shadow: var(--soft-shadow);
      margin-bottom: 18px;
    }

    .status-banner.critical,
    .status-banner.error {
      background: linear-gradient(90deg, var(--critical-soft), rgba(255, 255, 255, 0.94));
      border-color: #fecaca;
    }

    .status-banner.warning {
      background: linear-gradient(90deg, var(--warning-soft), rgba(255, 255, 255, 0.94));
      border-color: #fde68a;
    }

    .status-icon {
      display: grid;
      place-items: center;
      width: 56px;
      height: 56px;
      color: white;
      background: var(--blue-700);
      border-radius: 8px;
      box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
    }

    .critical .status-icon,
    .error .status-icon { background: var(--critical); box-shadow: 0 10px 24px rgba(220, 38, 38, 0.22); }
    .warning .status-icon { background: var(--warning); box-shadow: 0 10px 24px rgba(245, 158, 11, 0.22); }
    .ok .status-icon { background: var(--ok); box-shadow: 0 10px 24px rgba(22, 163, 74, 0.18); }

    .status-icon svg { width: 30px; height: 30px; }
    .status-copy span {
      display: block;
      margin-bottom: 4px;
      color: var(--slate-600);
      font-size: 13px;
    }
    .status-copy strong {
      display: block;
      color: var(--slate-950);
      font-size: clamp(20px, 1.8vw, 28px);
      line-height: 1.1;
    }

    .threshold-legend {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 12px;
      color: var(--slate-700);
      white-space: nowrap;
    }

    .legend-dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      border-radius: 999px;
      background: var(--blue-500);
    }
    .legend-dot.warning { background: var(--warning); }
    .legend-dot.critical { background: var(--critical); }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(156px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }

    .overview-kpi-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }

    .kpi-card,
    .card,
    .table-panel {
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid var(--slate-200);
      border-radius: var(--radius);
      box-shadow: var(--soft-shadow);
    }

    .kpi-card {
      position: relative;
      min-height: 126px;
      padding: 16px;
      overflow: hidden;
    }

    .overview-kpi-grid .kpi-card {
      min-height: 148px;
      padding: 18px 18px 16px;
    }

    .overview-kpi-grid .kpi-card::before {
      position: absolute;
      top: 0;
      right: 0;
      left: 0;
      height: 3px;
      content: "";
      background: var(--blue-600);
    }

    .overview-kpi-grid .kpi-card.ok::before { background: var(--ok); }
    .overview-kpi-grid .kpi-card.warning::before { background: var(--warning); }
    .overview-kpi-grid .kpi-card.critical::before,
    .overview-kpi-grid .kpi-card.error::before { background: var(--critical); }

    .overview-kpi-grid .kpi-card:nth-child(1),
    .overview-kpi-grid .kpi-card:nth-child(2),
    .overview-kpi-grid .kpi-card:nth-child(3) {
      background: linear-gradient(180deg, #ffffff, #f8fbff);
    }

    .overview-kpi-grid .kpi-card:nth-child(4),
    .overview-kpi-grid .kpi-card:nth-child(5),
    .overview-kpi-grid .kpi-card:nth-child(6) {
      background: linear-gradient(180deg, #ffffff, #fbfdff);
    }

    .overview-kpi-grid .kpi-card:nth-child(8),
    .overview-kpi-grid .kpi-card:nth-child(9) {
      background: linear-gradient(180deg, #ffffff, #fff8f8);
      border-color: #fecaca;
    }

    .kpi-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 16px;
    }

    .kpi-label {
      color: var(--slate-600);
      font-size: 12px;
      font-weight: 720;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }

    .kpi-value {
      display: block;
      margin-bottom: 10px;
      color: var(--slate-950);
      font-size: clamp(24px, 2.3vw, 34px);
      font-weight: 780;
      line-height: 1;
    }

    .overview-kpi-grid .kpi-value {
      margin-bottom: 14px;
      font-size: clamp(30px, 3vw, 42px);
      letter-spacing: 0;
    }

    .kpi-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--slate-500);
      font-size: 12px;
    }

    .kpi-meta span {
      display: inline-flex;
      gap: 4px;
      align-items: baseline;
    }

    .kpi-meta b {
      color: var(--slate-500);
      font-weight: 680;
    }

    .overview-kpi-grid .kpi-meta {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      padding-top: 12px;
      border-top: 1px solid var(--slate-100);
    }

    .overview-kpi-grid .kpi-meta span {
      display: grid;
      gap: 3px;
      min-width: 0;
      color: var(--slate-950);
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }

    .overview-kpi-grid .kpi-meta b {
      color: var(--slate-500);
      font-size: 11px;
      font-weight: 650;
      text-transform: uppercase;
    }

    .overview-grid,
    .panel-grid,
    .chart-grid {
      display: grid;
      gap: 14px;
    }

    .overview-grid {
      grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
    }

    .two-col {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .calibration-grid,
    .segments-grid {
      grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.6fr);
      align-items: stretch;
    }

    .stability-grid {
      grid-template-columns: minmax(360px, 0.95fr) repeat(2, minmax(320px, 1fr));
    }

    .data-grid {
      grid-template-columns: minmax(0, 1fr) minmax(320px, 0.8fr);
    }

    .card {
      min-width: 0;
      padding: 16px;
    }

    .card-title,
    .table-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }

    .card-title span,
    .table-title span {
      color: var(--slate-500);
      font-size: 12px;
    }

    .findings {
      display: grid;
      gap: 10px;
      padding: 0;
      margin: 18px 0;
      list-style: none;
    }

    .findings li,
    .metric-delta-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      padding: 12px;
      background: var(--slate-50);
      border: 1px solid var(--slate-200);
      border-radius: 8px;
    }

    .finding-marker {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--blue-600);
    }
    .finding-marker.critical,
    .finding-marker.error { background: var(--critical); }
    .finding-marker.warning { background: var(--warning); }
    .finding-marker.ok { background: var(--ok); }

    .finding-name {
      display: block;
      margin-bottom: 2px;
      color: var(--slate-950);
      font-weight: 720;
      overflow-wrap: anywhere;
    }

    .finding-message {
      display: block;
      color: var(--slate-500);
      font-size: 12px;
      overflow-wrap: anywhere;
    }

    .disclaimer {
      margin-bottom: 0;
      color: var(--slate-500);
      font-size: 12px;
    }

    .cover-list {
      display: grid;
      gap: 8px;
      margin: 0;
    }

    .cover-list div {
      display: grid;
      grid-template-columns: minmax(120px, 0.9fr) minmax(0, 1.1fr);
      gap: 12px;
      align-items: center;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--slate-100);
    }

    .cover-list dt {
      color: var(--slate-500);
      font-size: 12px;
    }

    .cover-list dd {
      min-width: 0;
      margin: 0;
      color: var(--slate-950);
      font-weight: 650;
      overflow-wrap: anywhere;
    }

    .chart-card {
      padding: 0;
      overflow: hidden;
    }

    .chart {
      min-height: 340px;
      padding: 8px 10px 4px;
    }

    .chart-grid .chart {
      background: white;
      border: 1px solid var(--slate-200);
      border-radius: var(--radius);
      box-shadow: var(--soft-shadow);
    }

    .metric-delta-list {
      display: grid;
      gap: 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .metric-delta-row {
      grid-template-columns: minmax(0, 1fr) auto;
    }

    .metric-delta-row span {
      color: var(--slate-500);
      font-size: 12px;
    }

    .metric-delta-row strong {
      display: block;
      color: var(--slate-950);
      font-size: 15px;
      line-height: 1.3;
    }

    .psi-bars {
      display: grid;
      gap: 16px;
      padding: 8px 2px 0;
    }

    .psi-row {
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr) 72px;
      gap: 12px;
      align-items: center;
    }

    .psi-name {
      color: var(--slate-950);
      font-weight: 680;
      overflow-wrap: anywhere;
    }

    .psi-type {
      display: block;
      color: var(--slate-500);
      font-size: 11px;
      font-weight: 500;
    }

    .psi-track {
      position: relative;
      height: 28px;
      overflow: hidden;
      background: var(--slate-100);
      border-radius: 5px;
    }

    .psi-fill {
      position: absolute;
      inset: 0 auto 0 0;
      min-width: 3px;
      background: linear-gradient(90deg, #f97316, var(--critical));
      border-radius: 5px;
    }

    .psi-value {
      color: var(--slate-950);
      font-variant-numeric: tabular-nums;
    }

    .dist-table {
      display: grid;
      gap: 8px;
      font-size: 12px;
    }

    .dist-head,
    .dist-row,
    .dist-total {
      display: grid;
      grid-template-columns: minmax(86px, 1fr) minmax(90px, 1fr) minmax(90px, 1fr);
      gap: 10px;
      align-items: center;
    }

    .dist-head {
      color: var(--slate-500);
      font-weight: 720;
    }

    .dist-row {
      min-height: 24px;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--slate-100);
    }

    .dist-total {
      padding-top: 2px;
      color: var(--slate-950);
      font-weight: 760;
    }

    .mini-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      justify-content: flex-end;
      color: var(--slate-700);
      font-variant-numeric: tabular-nums;
    }

    .mini-track {
      position: relative;
      width: min(72px, 45%);
      height: 8px;
      overflow: hidden;
      background: var(--slate-100);
      border-radius: 999px;
    }

    .mini-fill {
      position: absolute;
      inset: 0 auto 0 0;
      min-width: 2px;
      background: #9dc5ee;
      border-radius: 999px;
    }

    .mini-fill.current {
      background: var(--navy-800);
    }

    .table-panel {
      min-width: 0;
      margin-top: 14px;
      overflow: hidden;
    }

    .table-title {
      padding: 16px 18px 0;
      margin-bottom: 12px;
    }

    .table-wrap {
      width: 100%;
      overflow-x: auto;
      padding: 0 10px 10px;
    }

    table {
      width: 100%;
      min-width: 720px;
      border-collapse: collapse;
      font-size: 13px;
    }

    th,
    td {
      padding: 11px 12px;
      border-bottom: 1px solid var(--slate-100);
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }

    th {
      color: var(--navy-900);
      background: linear-gradient(180deg, #f8fbff, #f1f6fd);
      font-size: 12px;
      font-weight: 760;
    }

    td {
      color: var(--slate-950);
      font-variant-numeric: tabular-nums;
    }

    td:first-child,
    th:first-child { border-left: 0; }

    tbody tr:hover td { background: var(--blue-50); }

    .compact table { min-width: 520px; }
    .segment-drift-panel table {
      min-width: 1080px;
      font-size: 12px;
    }
    .segment-drift-panel th,
    .segment-drift-panel td {
      padding: 10px 11px;
    }
    .segment-drift-panel tbody tr:last-child td {
      color: var(--slate-950);
      background: #f8fbff;
      font-weight: 760;
      border-top: 1px solid var(--slate-200);
    }
    .status-cell {
      font-weight: 760;
    }
    .status-cell.ok { color: var(--ok); }
    .status-cell.warning { color: var(--warning); }
    .status-cell.critical,
    .status-cell.error { color: var(--critical); }
    .negative { color: #16a34a; }
    .positive { color: var(--critical); }
    .table-note {
      padding: 0 18px 16px;
      margin: 0;
      color: var(--slate-500);
      font-size: 12px;
    }

    .method-card {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }

    .method-card .card {
      min-height: 150px;
    }

    @media (max-width: 1180px) {
      .overview-kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .overview-grid,
      .calibration-grid,
      .segments-grid,
      .stability-grid,
      .data-grid,
      .two-col {
        grid-template-columns: 1fr;
      }
      .status-banner {
        grid-template-columns: auto minmax(0, 1fr);
      }
      .threshold-legend {
        grid-column: 1 / -1;
        justify-content: flex-start;
      }
      .method-card {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 760px) {
      .app-shell {
        display: block;
        padding: 0;
      }
      .sidebar {
        position: sticky;
        top: 0;
        z-index: 10;
        flex-direction: row;
        width: 100%;
        height: auto;
        padding: 10px 12px;
        overflow-x: auto;
      }
      .side-nav {
        flex-direction: row;
      }
      .brand-mark,
      .side-link {
        width: 42px;
        height: 42px;
        flex: 0 0 42px;
      }
      .report-window {
        border-radius: 0;
        border-right: 0;
        border-left: 0;
      }
      .topbar {
        align-items: flex-start;
        flex-direction: column;
        padding: 22px 18px 16px;
      }
      .header-actions {
        width: 100%;
        justify-content: stretch;
      }
      .header-actions button {
        justify-content: center;
        flex: 1 1 150px;
      }
      .tabs {
        padding: 0 12px;
      }
      .tab {
        min-height: 58px;
        padding: 0 12px;
      }
      main {
        padding: 16px;
      }
      .section-heading,
      .status-banner {
        align-items: stretch;
        grid-template-columns: 1fr;
        flex-direction: column;
      }
      .status-icon {
        width: 48px;
        height: 48px;
      }
      .overview-kpi-grid {
        grid-template-columns: 1fr;
      }
      .overview-kpi-grid .kpi-meta {
        grid-template-columns: repeat(3, minmax(64px, 1fr));
      }
      .psi-row {
        grid-template-columns: 1fr;
        gap: 7px;
      }
      .dist-head,
      .dist-row,
      .dist-total {
        grid-template-columns: minmax(70px, 1fr) minmax(82px, 1fr) minmax(82px, 1fr);
        gap: 6px;
      }
      table {
        min-width: 640px;
      }
    }
    """


def _default_view(result: "PDValidationResult") -> str:
    if any(
        name.startswith("psi_") and metric.status.value in {"CRITICAL", "ERROR", "WARNING"}
        for name, metric in result.metrics.items()
    ):
        return "stability"
    return "overview"


def _view_attrs(view: str, default_view: str) -> str:
    active = view == default_view
    hidden = "" if active else " hidden"
    active_class = " active" if active else ""
    return f'class="view-section{active_class}" data-view="{escape(view)}"{hidden}'


def _sidebar(default_view: str, *, include_methodology: bool) -> str:
    items = [
        ("overview", "grid"),
        ("discrimination", "trend"),
        ("calibration", "ruler"),
        ("stability", "pulse"),
        ("segments", "users"),
        ("data-quality", "shield"),
        ("tables", "table"),
    ]
    if include_methodology:
        items.insert(-1, ("methodology", "book"))
    links = "\n".join(
        f'<a class="side-link {"active" if anchor == default_view else ""}" href="#{anchor}" data-view-target="{anchor}" aria-label="{anchor}">{_icon(icon)}</a>'
        for anchor, icon in items
    )
    return f"""
    <aside class="sidebar" aria-label="Report navigation">
      <a class="brand-mark" href="#overview" data-view-target="overview" aria-label="Report home">{_icon("bars")}</a>
      <nav class="side-nav">{links}</nav>
      <a class="side-link" href="#tables" data-view-target="tables" aria-label="Export tables">{_icon("download")}</a>
    </aside>
    """


def _tabs(locale: dict[str, str], default_view: str, *, include_methodology: bool) -> str:
    tabs = [
        ("overview", "grid", "Overview"),
        ("discrimination", "trend", t(locale, "discrimination_title")),
        ("calibration", "ruler", t(locale, "calibration_title")),
        ("stability", "pulse", t(locale, "stability_title")),
        ("segments", "users", "Segments"),
        ("data-quality", "shield", t(locale, "data_quality")),
    ]
    if include_methodology:
        tabs.append(("methodology", "book", t(locale, "methodology_title")))
    return (
        '<nav class="tabs" aria-label="Report sections">'
        + "\n".join(
            f'<a class="tab {"active" if anchor == default_view else ""}" href="#{anchor}" data-view-target="{anchor}">{_icon(icon)}<span>{escape(label)}</span></a>'
            for anchor, icon, label in tabs
        )
        + "</nav>"
    )


def _status_banner(status: str, locale: dict[str, str]) -> str:
    status_class = _status_class(status)
    if status == "CRITICAL":
        message = "One or more validation metrics breached a critical threshold."
    elif status == "WARNING":
        message = "One or more validation metrics require review."
    elif status == "OK":
        message = "All available validation evidence is within configured thresholds."
    else:
        message = "Some validation evidence is unavailable or requires follow-up."
    return f"""
    <div class="status-banner {status_class}">
      <div class="status-icon">{_icon("alert" if status in {"CRITICAL", "ERROR"} else "check")}</div>
      <div class="status-copy">
        <span>{t(locale, "overall_status")}</span>
        <strong>{escape(status)}</strong>
        <p>{escape(message)}</p>
      </div>
      <div class="threshold-legend">
        <span><i class="legend-dot warning"></i>{t(locale, "threshold_warning")} 0.10</span>
        <span><i class="legend-dot critical"></i>{t(locale, "threshold_critical")} 0.25</span>
      </div>
    </div>
    """


def _stability_banner(result: "PDValidationResult") -> str:
    psi_statuses = [
        metric.status.value
        for name, metric in result.metrics.items()
        if name.startswith("psi_") and metric.status.value in {"WARNING", "CRITICAL", "ERROR"}
    ]
    status = "CRITICAL" if "CRITICAL" in psi_statuses else "WARNING" if psi_statuses else "OK"
    if status == "CRITICAL":
        message = "One or more stability metrics have breached the critical threshold."
    elif status == "WARNING":
        message = "One or more stability metrics are above the warning threshold."
    else:
        message = "Stability metrics are within the configured threshold."
    return f"""
    <div class="status-banner {_status_class(status)}">
      <div class="status-icon">{_icon("alert" if status != "OK" else "check")}</div>
      <div class="status-copy">
        <span>Stability Status</span>
        <strong>{escape(status)}</strong>
        <p>{escape(message)}</p>
      </div>
      <div class="threshold-legend">
        <span><i class="legend-dot warning"></i>Warning 0.10</span>
        <span><i class="legend-dot critical"></i>Critical 0.25</span>
      </div>
    </div>
    """


def _kpi_grid(
    result: "PDValidationResult",
    metrics: list[tuple[str, str]],
    *,
    class_name: str = "",
) -> str:
    cards = "\n".join(_kpi_card(result.metrics.get(name), label) for name, label in metrics)
    classes = " ".join(name for name in ["kpi-grid", class_name] if name)
    return f'<div class="{escape(classes)}">{cards}</div>'


def _kpi_card(metric: "MetricResult | None", label: str) -> str:
    status = metric.status.value if metric else "NOT_APPLICABLE"
    delta = metric.delta if metric else None
    metric_name = metric.name if metric else None
    meta = []
    if metric and metric.reference_value is not None:
        meta.append(("Ref", _format_metric_value(metric.reference_value, metric_name)))
    if metric and metric.current_value is not None:
        meta.append(("Cur", _format_metric_value(metric.current_value, metric_name)))
    if delta is not None:
        meta.append(("Delta", _format_metric_value(delta, metric_name)))
    meta_html = (
        "".join(
            f"<span><b>{escape(label_text)}</b>{escape(value_text)}</span>"
            for label_text, value_text in meta
        )
        or "<span>No comparison</span>"
    )
    return f"""
    <article class="kpi-card {_status_class(status)}">
      <div class="kpi-top">
        <span class="kpi-label">{escape(label)}</span>
        <span class="status-pill {_status_class(status)}">{escape(status)}</span>
      </div>
      <strong class="kpi-value">{_format_metric_value(metric.value if metric else None, metric_name) or "-"}</strong>
      <div class="kpi-meta">{meta_html}</div>
    </article>
    """


def _findings_list(result: "PDValidationResult") -> str:
    findings: list[tuple[str, str, str]] = []
    for metric in result.metrics.values():
        if metric.status.value in {"CRITICAL", "WARNING", "ERROR"}:
            findings.append(
                (metric.status.value, metric.name, metric.message or "Threshold review needed")
            )
    for check in result.checks:
        if check.status.value in {"CRITICAL", "WARNING", "ERROR"}:
            findings.append((check.status.value, check.name, check.message))
    if not findings:
        findings.append(
            ("OK", "Validation evidence", "No critical or warning findings were reported.")
        )
    rows = "\n".join(
        f"""
        <li>
          <i class="finding-marker {_status_class(status)}"></i>
          <span><strong class="finding-name">{escape(name)}</strong><span class="finding-message">{escape(message)}</span></span>
          <span class="status-pill {_status_class(status)}">{escape(status)}</span>
        </li>
        """
        for status, name, message in findings[:6]
    )
    return f'<ul class="findings">{rows}</ul>'


def _metric_delta_list(result: "PDValidationResult", metric_names: list[str]) -> str:
    rows = []
    for name in metric_names:
        metric = result.metrics.get(name)
        if metric is None:
            continue
        rows.append(
            f"""
            <li class="metric-delta-row">
              <span>
                <strong>{escape(metric.name)}</strong>
                <span>Reference {_format_metric_value(metric.reference_value, metric.name) or "-"} / Current {_format_metric_value(metric.current_value, metric.name) or "-"}</span>
              </span>
              <span class="status-pill {_status_class(metric.status.value)}">{escape(metric.status.value)}</span>
            </li>
            """
        )
    return f'<ul class="metric-delta-list">{"".join(rows)}</ul>'


def _psi_variable_card(result: "PDValidationResult", locale: dict[str, str]) -> str:
    table = result.tables.get("psi_by_variable")
    if table is None or table.is_empty():
        return _empty_card("PSI by Variable", "No PSI variable summary was generated.")
    rows = sorted(table.to_dicts(), key=lambda row: float(row.get("psi") or 0), reverse=True)
    max_value = max([float(row.get("psi") or 0) for row in rows] + [0.25])
    body = "\n".join(
        f"""
        <div class="psi-row">
          <div class="psi-name">{escape(str(row.get("variable", "")))}<span class="psi-type">{escape(str(row.get("variable_type", "")))}</span></div>
          <div class="psi-track"><span class="psi-fill" style="width: {_bar_width(row.get("psi"), max_value)}%;"></span></div>
          <div class="psi-value">{_format_metric_value(_float_or_none(row.get("psi")), "psi")}</div>
        </div>
        """
        for row in rows
    )
    return f"""
    <article class="card">
      <div class="card-title">
        <h3>PSI by Variable</h3>
        <span>{t(locale, "threshold_critical")} 0.25</span>
      </div>
      <div class="psi-bars">{body}</div>
    </article>
    """


def _distribution_card(
    result: "PDValidationResult", table_name: str, title: str, label: str
) -> str:
    table = result.tables.get(table_name)
    if table is None or table.is_empty():
        return _empty_card(title, f"No {label} distribution comparison was generated.")
    rows = table.to_dicts()
    key = "bin" if "bin" in rows[0] else "category"
    total_ref = sum(float(row.get("reference_share") or 0) for row in rows)
    total_cur = sum(float(row.get("current_share") or 0) for row in rows)
    body = "\n".join(
        f"""
        <div class="dist-row">
          <span>{escape(str(row.get(key, "")))}</span>
          {_mini_bar(row.get("reference_share"), "reference")}
          {_mini_bar(row.get("current_share"), "current")}
        </div>
        """
        for row in rows[:10]
    )
    return f"""
    <article class="card">
      <div class="card-title">
        <h3>{escape(title)}</h3>
        <span>Deciles</span>
      </div>
      <div class="dist-table">
        <div class="dist-head"><span>{escape(label)} Decile</span><span>Reference Share</span><span>Current Share</span></div>
        {body}
        <div class="dist-total"><span>Total</span><span>{_format_percent(total_ref, decimals=2)}</span><span>{_format_percent(total_cur, decimals=2)}</span></div>
      </div>
    </article>
    """


def _mini_bar(value: object, kind: str) -> str:
    float_value = _float_or_none(value) or 0
    return f"""
    <span class="mini-bar">
      <span class="mini-track"><span class="mini-fill {escape(kind)}" style="width: {_bar_width(float_value, 1)}%;"></span></span>
      <span>{_format_percent(float_value, decimals=2)}</span>
    </span>
    """


def _segment_summary_card(result: "PDValidationResult") -> str:
    table = result.tables.get("segment_metrics")
    if table is None or table.is_empty():
        return _empty_card("Segment Calibration", "No configured segment metrics were generated.")
    rows = table.to_dicts()
    ok_count = sum(1 for row in rows if row.get("status") == "OK")
    insufficient_count = sum(1 for row in rows if row.get("status") == "INSUFFICIENT_DATA")
    return f"""
    <article class="card insight-card">
      <div class="card-title"><h3>Segment Calibration</h3><span>{len(rows)} segments</span></div>
      <ul class="metric-delta-list">
        <li class="metric-delta-row"><span><strong>{ok_count}</strong><span>segments with full metrics</span></span><span class="status-pill ok">OK</span></li>
        <li class="metric-delta-row"><span><strong>{insufficient_count}</strong><span>segments with limited evidence</span></span><span class="status-pill insufficient_data">INSUFFICIENT</span></li>
      </ul>
    </article>
    """


def _segment_drift_panel(result: "PDValidationResult", locale: dict[str, str]) -> str:
    table = result.tables.get("segment_drift")
    if table is None or table.is_empty():
        return ""
    source_rows = table.to_dicts()
    rows = [_segment_drift_row(row) for row in source_rows]
    total = _segment_drift_total(source_rows)
    if total:
        rows.append(total)
    headers = [
        "Segment",
        "Reference Share",
        "Current Share",
        "Population PSI",
        "Reference Bad Rate",
        "Current Bad Rate",
        "Bad Rate Delta",
        "Reference Avg PD",
        "Current Avg PD",
        "Avg PD Delta",
    ]
    body = "\n".join(
        "<tr>"
        + "".join(
            _table_cell(header.lower().replace(" ", "_"), row.get(header, "-"))
            for header in headers
        )
        + "</tr>"
        for row in rows
    )
    header_html = "".join(f"<th>{escape(header)}</th>" for header in headers)
    return f"""
    <article class="table-panel segment-drift-panel">
      <div class="table-title">
        <h3>Segment Drift</h3>
        <span>{len(source_rows)} rows</span>
      </div>
      <div class="table-wrap">
        <table><thead><tr>{header_html}</tr></thead><tbody>{body}</tbody></table>
      </div>
    </article>
    """


def _segment_drift_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "Segment": row.get("segment"),
        "Reference Share": _format_percent(_float_or_none(row.get("reference_share"))),
        "Current Share": _format_percent(_float_or_none(row.get("current_share"))),
        "Population PSI": _format_metric_value(_float_or_none(row.get("population_psi")), "psi")
        or "-",
        "Reference Bad Rate": _format_percent(_float_or_none(row.get("reference_bad_rate"))),
        "Current Bad Rate": _format_percent(_float_or_none(row.get("current_bad_rate"))),
        "Bad Rate Delta": _format_percent(_float_or_none(row.get("bad_rate_delta"))),
        "Reference Avg PD": _format_percent(_float_or_none(row.get("reference_avg_pd"))),
        "Current Avg PD": _format_percent(_float_or_none(row.get("current_avg_pd"))),
        "Avg PD Delta": _format_percent(_float_or_none(row.get("avg_pd_delta"))),
    }


def _segment_drift_total(rows: list[dict[str, object]]) -> dict[str, object] | None:
    reference_count = sum(_float_or_none(row.get("reference_count")) or 0 for row in rows)
    current_count = sum(_float_or_none(row.get("current_count")) or 0 for row in rows)
    if reference_count <= 0 and current_count <= 0:
        return None
    ref_bad_events = sum(
        (
            (_float_or_none(row.get("reference_bad_rate")) or 0)
            * (_float_or_none(row.get("reference_count")) or 0)
        )
        for row in rows
    )
    cur_bad_events = sum(
        (
            (_float_or_none(row.get("current_bad_rate")) or 0)
            * (_float_or_none(row.get("current_count")) or 0)
        )
        for row in rows
    )
    ref_pd_total = sum(
        (
            (_float_or_none(row.get("reference_avg_pd")) or 0)
            * (_float_or_none(row.get("reference_count")) or 0)
        )
        for row in rows
    )
    cur_pd_total = sum(
        (
            (_float_or_none(row.get("current_avg_pd")) or 0)
            * (_float_or_none(row.get("current_count")) or 0)
        )
        for row in rows
    )
    ref_bad_rate = ref_bad_events / reference_count if reference_count else None
    cur_bad_rate = cur_bad_events / current_count if current_count else None
    ref_avg_pd = ref_pd_total / reference_count if reference_count else None
    cur_avg_pd = cur_pd_total / current_count if current_count else None
    return {
        "Segment": "Total",
        "Reference Share": _format_percent(1.0 if reference_count else None, decimals=2),
        "Current Share": _format_percent(1.0 if current_count else None, decimals=2),
        "Population PSI": _format_metric_value(
            sum(_float_or_none(row.get("population_psi")) or 0 for row in rows), "psi"
        ),
        "Reference Bad Rate": _format_percent(ref_bad_rate, decimals=2),
        "Current Bad Rate": _format_percent(cur_bad_rate, decimals=2),
        "Bad Rate Delta": _format_percent(
            cur_bad_rate - ref_bad_rate
            if cur_bad_rate is not None and ref_bad_rate is not None
            else None,
            decimals=2,
        ),
        "Reference Avg PD": _format_percent(ref_avg_pd, decimals=2),
        "Current Avg PD": _format_percent(cur_avg_pd, decimals=2),
        "Avg PD Delta": _format_percent(
            cur_avg_pd - ref_avg_pd if cur_avg_pd is not None and ref_avg_pd is not None else None,
            decimals=2,
        ),
    }


def _checks_panel(result: "PDValidationResult", locale: dict[str, str]) -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td>{escape(check.name)}</td>
          <td class="status-cell {_status_class(check.status.value)}">{escape(check.status.value)}</td>
          <td>{escape(check.message)}</td>
        </tr>
        """
        for check in result.checks
    )
    return f"""
    <article class="table-panel">
      <div class="table-title"><h3>{t(locale, "data_quality")}</h3><span>{len(result.checks)} checks</span></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>{t(locale, "check")}</th><th>{t(locale, "status")}</th><th>{t(locale, "message")}</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </article>
    """


def _model_card_section(result: "PDValidationResult", locale: dict[str, str]) -> str:
    if not result.config.report.include_model_card:
        return ""
    model = result.config.model
    body = escape(
        t(locale, "model_card_body").format(
            purpose=str(model.purpose or t(locale, "not_specified")),
            owner=str(model.owner or t(locale, "not_specified")),
        )
    )
    return f"""
    <section id="model-card" class="view-section" data-view="methodology" hidden>
      <div class="section-heading">
        <div>
          <p class="eyebrow">Governance</p>
          <h2>{t(locale, "model_card")}</h2>
          <p>{body}</p>
        </div>
      </div>
    </section>
    """


def _scripts(default_view: str) -> str:
    return """
    (() => {
      const controls = Array.from(document.querySelectorAll("[data-view-target]"));
      const sections = Array.from(document.querySelectorAll("[data-view]"));
      const defaultView = "__DEFAULT_VIEW__";
      const validViews = new Set(sections.map((section) => section.dataset.view));

      function setActiveView(view, options = {}) {
        const nextView = validViews.has(view) ? view : defaultView;
        sections.forEach((section) => {
          const isActive = section.dataset.view === nextView;
          section.hidden = !isActive;
          section.classList.toggle("active", isActive);
        });
        controls.forEach((control) => {
          control.classList.toggle("active", control.dataset.viewTarget === nextView);
        });
        if (!options.skipHash) {
          history.replaceState(null, "", `#${nextView}`);
        }
        if (!options.preserveScroll) {
          window.scrollTo({ top: 0, left: 0, behavior: "instant" });
        }
        window.dispatchEvent(new Event("resize"));
      }

      controls.forEach((control) => {
        control.addEventListener("click", (event) => {
          event.preventDefault();
          setActiveView(control.dataset.viewTarget || defaultView);
        });
      });

      const initialView = window.location.hash.replace("#", "");
      setActiveView(initialView || defaultView, { skipHash: !initialView, preserveScroll: true });
    })();
    """.replace("__DEFAULT_VIEW__", escape(default_view))


def _methodology_section(result: "PDValidationResult", locale: dict[str, str]) -> str:
    if not result.config.report.include_methodology:
        return ""
    return f"""
    <section id="methodology" class="view-section" data-view="methodology" hidden>
      <div class="section-heading">
        <div>
          <p class="eyebrow">Evidence framework</p>
          <h2>{t(locale, "methodology_title")}</h2>
          <p>{t(locale, "methodology_body")}</p>
        </div>
      </div>
      <div class="method-card">
        <article class="card"><div class="card-title"><h3>{t(locale, "discrimination_title")}</h3></div><p>AUC, Gini, KS, lift, bad-rate and capture-rate outputs are reported as aggregate metrics and tables.</p></article>
        <article class="card"><div class="card-title"><h3>{t(locale, "calibration_title")}</h3></div><p>Brier Score, Log Loss, ECE, O/E ratio and calibration bins are reported when calculable.</p></article>
        <article class="card"><div class="card-title"><h3>{t(locale, "stability_title")}</h3></div><p>PSI summarizes distribution stability between reference and current samples.</p></article>
      </div>
    </section>
    """


def _table_panel(
    name: str,
    table: pl.DataFrame | None,
    locale: dict[str, str],
    *,
    compact: bool = False,
) -> str:
    if table is None or table.is_empty():
        return ""
    return _table_section(name, table.to_dicts(), locale, compact=compact)


def _table_section(
    name: str,
    rows: list[dict[str, object]],
    locale: dict[str, str],
    *,
    compact: bool = False,
) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    header_html = "".join(
        f"<th>{escape(_labelize_header(str(header), locale))}</th>" for header in headers
    )
    body_html = "\n".join(
        "<tr>" + "".join(_table_cell(header, row.get(header)) for header in headers) + "</tr>"
        for row in rows[:200]
    )
    suffix = f'<p class="table-note">{t(locale, "table_truncated")}</p>' if len(rows) > 200 else ""
    return f"""
    <article class="table-panel {"compact" if compact else ""}">
      <div class="table-title">
        <h3>{escape(name.replace("_", " ").title())}</h3>
        <span>{len(rows)} rows</span>
      </div>
      <div class="table-wrap">
        <table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>
      </div>
      {suffix}
    </article>
    """


def _table_cell(header: str, value: object) -> str:
    class_names: list[str] = []
    lower_header = header.lower()
    if lower_header == "status":
        class_names.append(f"status-cell {_status_class(str(value))}")
    float_value = _float_or_none(value)
    if float_value is not None and lower_header.endswith("delta"):
        class_names.append("negative" if float_value < 0 else "positive" if float_value > 0 else "")
    class_attr = f' class="{" ".join(name for name in class_names if name)}"' if class_names else ""
    return f"<td{class_attr}>{escape(_format_cell(value, header=header))}</td>"


def _labelize_header(header: str, locale: dict[str, str]) -> str:
    if header in locale:
        return t(locale, header)
    custom = {
        "threshold_warning": t(locale, "threshold_warning"),
        "threshold_critical": t(locale, "threshold_critical"),
        "reference_value": t(locale, "reference_value"),
        "current_value": t(locale, "current_value"),
        "delta": t(locale, "delta"),
        "status": t(locale, "status"),
        "message": t(locale, "message"),
        "metric": t(locale, "metric"),
        "name": t(locale, "metric"),
    }
    if header in custom:
        return custom[header]
    return header.replace("_", " ").title()


def _empty_card(title: str, message: str) -> str:
    return f"""
    <article class="card">
      <div class="card-title"><h3>{escape(title)}</h3></div>
      <p>{escape(message)}</p>
    </article>
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


def _format_metric_value(value: float | None, metric_name: str | None = None) -> str:
    if value is None:
        return ""
    metric = (metric_name or "").lower()
    if metric in {"auc", "gini", "ks", "brier", "log_loss", "ece", "oe_ratio"}:
        return escape(f"{value:.2f}")
    if metric.startswith("psi") or metric == "psi":
        return escape(f"{value:.2f}")
    return escape(f"{value:.2f}")


def _format_cell(value: object, *, header: str = "") -> str:
    if value is None:
        return "-"
    lower_header = header.lower()
    numeric_value = _float_or_none(value)
    if numeric_value is not None:
        if any(token in lower_header for token in ["rate", "share", "capture"]) or lower_header in {
            "mean_pd",
            "avg_pd",
            "pd_min",
            "pd_max",
            "observed_rate",
            "event_rate",
            "oe_ratio",
        }:
            return _format_percent(numeric_value, decimals=2)
        if "score" in lower_header:
            return f"{numeric_value:.2f}"
        if lower_header in {
            "count",
            "weighted_count",
            "events",
            "non_events",
            "n",
            "rows",
            "columns",
            "bin",
        }:
            return f"{numeric_value:.0f}"
        if lower_header in {
            "auc",
            "gini",
            "ks",
            "brier",
            "log_loss",
            "ece",
            "psi",
            "population_psi",
            "lift",
        }:
            return f"{numeric_value:.2f}"
        if lower_header.endswith("value") or lower_header in {
            "delta",
            "threshold_warning",
            "threshold_critical",
            "sample_size",
            "event_count",
            "non_event_count",
        }:
            return f"{numeric_value:.2f}"
        if isinstance(value, float):
            return f"{numeric_value:.2f}"
    return str(value)


def _format_percent(value: float | None, *, decimals: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{decimals}f}%"


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bar_width(value: object, max_value: float) -> str:
    float_value = _float_or_none(value) or 0
    if max_value <= 0:
        return "0"
    return f"{max(0, min(100, (float_value / max_value) * 100)):.2f}"


def _status_class(status: str) -> str:
    return status.lower().replace(" ", "_")


def _short_datetime(value: str) -> str:
    return value.replace("T", " ")[:16]


def _style_figure(figure: go.Figure, *, height: int) -> go.Figure:
    figure.update_layout(
        font={"family": "Inter, Arial, sans-serif", "color": "#0f172a"},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        margin={"l": 54, "r": 24, "t": 58, "b": 72},
        height=height,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
        },
        colorway=["#1d4ed8", "#0f2f5f", "#93c5fd", "#f59e0b", "#dc2626"],
        title={"font": {"size": 16}, "x": 0.02, "xanchor": "left"},
    )
    figure.update_xaxes(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0")
    figure.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0")
    return figure


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
            line={"width": 3, "color": "#1d4ed8"},
            marker={"size": 7},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[row["bin"] for row in rows],
            y=[row["cumulative_population_share"] for row in rows],
            mode="lines+markers",
            name="Cumulative population share",
            line={"width": 3, "color": "#0f2f5f"},
            marker={"size": 7},
        )
    )
    figure.update_layout(
        title=t(locale, "lift_capture_title"),
        xaxis_title=t(locale, "risk_bin"),
        yaxis_title=t(locale, "share"),
        yaxis_tickformat=".0%",
    )
    return _style_figure(figure, height=360)


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
            marker={"color": "#1d4ed8", "line": {"color": "#1e40af", "width": 1}},
        )
    )
    figure.update_layout(
        title=t(locale, "bad_rate_by_risk_bin"),
        xaxis_title=t(locale, "risk_bin"),
        yaxis_title="Bad rate",
        yaxis_tickformat=".0%",
    )
    return _style_figure(figure, height=360)


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
            line={"width": 3, "color": "#1d4ed8"},
            marker={"size": 8, "color": "#1d4ed8"},
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
            line={"dash": "dash", "width": 2, "color": "#64748b"},
        )
    )
    figure.update_layout(
        title=t(locale, "calibration_chart_title"),
        xaxis_title=t(locale, "calibration_x"),
        yaxis_title=t(locale, "calibration_y"),
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
    )
    return _style_figure(figure, height=390)


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
    x_field = "bin" if "bin" in rows[0] else "category"
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[row[x_field] for row in rows],
            y=[row["reference_share"] for row in rows],
            name="Reference share",
            marker={"color": "#93c5fd"},
        )
    )
    figure.add_trace(
        go.Bar(
            x=[row[x_field] for row in rows],
            y=[row["current_share"] for row in rows],
            name="Current share",
            marker={"color": "#0f2f5f"},
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Bin",
        yaxis_title=t(locale, "population_share"),
        yaxis_tickformat=".0%",
        barmode="group",
    )
    return _style_figure(figure, height=360)


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
            marker={"color": "#1d4ed8"},
        )
    )
    figure.add_trace(
        go.Bar(
            x=[str(row["segment"]) for row in rows],
            y=[row.get("mean_pd") for row in rows],
            name="Mean PD",
            marker={"color": "#93c5fd"},
        )
    )
    figure.update_layout(
        title=t(locale, "segment_calibration_summary"),
        xaxis_title=t(locale, "segment_x"),
        yaxis_title="Rate",
        yaxis_tickformat=".0%",
        barmode="group",
    )
    return _style_figure(figure, height=390)


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


def _icon(name: str) -> str:
    icons = {
        "alert": '<svg viewBox="0 0 24 24"><path d="M12 4 3 20h18L12 4Z"/><path d="M12 9v5"/><path d="M12 18h.01"/></svg>',
        "bars": '<svg viewBox="0 0 24 24"><rect x="4" y="11" width="3" height="8" rx="1"/><rect x="10" y="5" width="3" height="14" rx="1"/><rect x="16" y="8" width="3" height="11" rx="1"/></svg>',
        "book": '<svg viewBox="0 0 24 24"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v17H6.5A2.5 2.5 0 0 1 4 17.5Z"/><path d="M4 17.5A2.5 2.5 0 0 0 6.5 20H20"/><path d="M8 7h8"/></svg>',
        "check": '<svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></svg>',
        "download": '<svg viewBox="0 0 24 24"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M4 19h16"/></svg>',
        "grid": '<svg viewBox="0 0 24 24"><rect x="4" y="4" width="6" height="6"/><rect x="14" y="4" width="6" height="6"/><rect x="4" y="14" width="6" height="6"/><rect x="14" y="14" width="6" height="6"/></svg>',
        "pulse": '<svg viewBox="0 0 24 24"><path d="M3 12h4l2-7 4 14 2-7h6"/></svg>',
        "ruler": '<svg viewBox="0 0 24 24"><path d="m4 17 10-10 3 3L7 20Z"/><path d="m13 8 2 2"/><path d="m10 11 2 2"/><path d="m7 14 2 2"/></svg>',
        "share": '<svg viewBox="0 0 24 24"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8.6 10.6 5.8-3.2"/><path d="m8.6 13.4 5.8 3.2"/></svg>',
        "shield": '<svg viewBox="0 0 24 24"><path d="M12 3 5 6v6c0 4.5 3 7.5 7 9 4-1.5 7-4.5 7-9V6Z"/><path d="m9 12 2 2 4-5"/></svg>',
        "table": '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 10h18"/><path d="M9 4v16"/><path d="M15 4v16"/></svg>',
        "trend": '<svg viewBox="0 0 24 24"><path d="M4 17 10 11l4 4 6-8"/><path d="M15 7h5v5"/></svg>',
        "users": '<svg viewBox="0 0 24 24"><path d="M16 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2"/><circle cx="9.5" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    }
    return icons.get(name, icons["grid"])
