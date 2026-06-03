# Examples

These examples use synthetic or prepared aggregate-safe data.

Run:

```bash
uv run python examples/01_basic_pd_validation.py
uv run python examples/02_generate_html_report.py
```

Notebook:

- `06_kaggle_training_validation_reports.ipynb`: downloads a public Kaggle credit
  dataset, trains Logistic Regression and XGBoost models, and generates one CRVK
  training-validation HTML report per model on the held-out test set.
