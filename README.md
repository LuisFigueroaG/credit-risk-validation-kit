# Credit Risk Validation Kit

[![CI](https://github.com/LuisFigueroaG/credit-risk-validation-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/LuisFigueroaG/credit-risk-validation-kit/actions/workflows/ci.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/LuisFigueroaG/credit-risk-validation-kit/badge)](https://securityscorecards.dev/viewer/?uri=github.com/LuisFigueroaG/credit-risk-validation-kit)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-not%20published-lightgrey.svg)](https://pypi.org/)

The missing validation and reporting layer for credit risk models.

Credit Risk Validation Kit (`crvk`) supports validation evidence for binary PD
models. It validates input contracts, computes discrimination, calibration and
stability metrics, exports aggregate tables, writes JSON metrics, and generates
self-contained HTML reports.

It does not approve models, certify regulatory compliance, calculate regulatory
capital, or calculate official provisions.

Generate a complete credit risk PD validation report in one command.

## Install

```bash
uv sync --all-extras --dev
```

## Python Quickstart

```python
import polars as pl
from credit_risk_validation import PDValidationSuite

reference_df = pl.read_parquet("data/processed/give_me_some_credit/reference.parquet")
current_df = pl.read_parquet("data/processed/give_me_some_credit/current.parquet")

suite = PDValidationSuite(
    target_col="target",
    pd_col="pd",
    score_col="score",
    segment_cols=["segment"],
)

result = suite.run(reference_data=reference_df, current_data=current_df)
result.to_html("reports/pd_validation_report.html")
result.to_json("reports/pd_validation_metrics.json")
result.to_tables("reports/tables")
```

## CLI Quickstart

```bash
uv run crvk pd-validate \
  --config examples/configs/give_me_some_credit.yml \
  --reference data/processed/give_me_some_credit/reference.parquet \
  --current data/processed/give_me_some_credit/current.parquet \
  --output-html reports/give_me_some_credit_report.html \
  --output-json reports/give_me_some_credit_metrics.json \
  --output-tables reports/give_me_some_credit_tables \
  --output-model-card reports/give_me_some_credit_model_card.md
```

## Metrics

- Discrimination: AUC ROC, Gini, KS, lift table, bad rate by bin, event capture.
- Calibration: Brier Score, Log Loss, calibration bins, ECE, MCE, O/E ratio,
  calibration-in-the-large, calibration slope.
- Stability: PSI for PD and score.
- Segments: aggregate discrimination and calibration summaries by configured
  segments when sample size is sufficient.

## Report Preview

The HTML report contains a cover, executive summary, data quality checks,
discrimination metrics, calibration diagnostics, stability summaries, segment
tables, methodology notes, and export references. It is self-contained and uses
aggregate evidence only.

## Datasets

Dataset harness commands use the Kaggle CLI. In this environment, Kaggle network
access requires escalated execution by the coding agent. Datasets that require
manual terms acceptance are skipped until the user accepts terms in Kaggle.

```bash
uv run crvk datasets download --dataset give_me_some_credit
uv run crvk datasets prepare --dataset give_me_some_credit --sample-size 5000
uv run crvk datasets run-harness --dataset give_me_some_credit
```

Raw downloads, processed data, ZIP files and generated reports are ignored by
git.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv run pytest --cov=credit_risk_validation --cov-report=term-missing
uv build
```

## Roadmap

- v0.1.0: binary PD validation MVP.
- v0.2.0: stronger calibration diagnostics.
- v0.3.0: richer segment analysis.
- v0.4.0: champion vs challenger reports.
- v1.0.0: stable API.

## License

Apache License 2.0. See [LICENSE](LICENSE).
