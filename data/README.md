# Data Directory

This directory is intentionally mostly ignored by git.

- `raw/`: original Kaggle downloads, ZIP files, CSVs or Parquet files.
- `processed/`: standardized reference/current splits and dataset metadata.

Do not commit raw datasets, processed datasets, customer data, generated model
outputs, or credentials. The dataset harness writes reproducibility metadata to
`data/processed/<dataset>/metadata.json`.
