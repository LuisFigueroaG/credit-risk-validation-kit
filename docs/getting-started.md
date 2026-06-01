# Getting Started

Install dependencies:

```bash
uv sync --all-extras --dev
```

Run a validation:

```bash
uv run crvk pd-validate --config examples/configs/give_me_some_credit.yml \
  --reference data/processed/give_me_some_credit/reference.parquet \
  --current data/processed/give_me_some_credit/current.parquet \
  --output-html reports/give_me_some_credit_report.html
```
