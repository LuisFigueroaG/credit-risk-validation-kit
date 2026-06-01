# Contributing

Thanks for considering a contribution to Credit Risk Validation Kit.

## Development

Use `uv` for dependency management:

```bash
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

Use short-lived branches from `master` and open pull requests back to `master`.
Do not commit raw datasets, generated reports, credentials, or customer data.

## Scope

The v0.x line supports validation evidence for binary PD models. It does not
certify regulatory compliance, approve models, calculate capital, or calculate
official provisions.
