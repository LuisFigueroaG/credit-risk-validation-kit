"""Markdown model card export."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from credit_risk_validation.results import PDValidationResult


def render_model_card(result: "PDValidationResult") -> str:
    """Renderiza model card en Markdown."""

    model = result.config.model
    metrics = "\n".join(
        f"- {metric.name}: {metric.value if metric.value is not None else 'not available'} ({metric.status.value})"
        for metric in result.metrics.values()
    )
    return f"""# Model Card: {model.name}

## Model Information

- Version: {model.version}
- Library version: {_metadata_value(result, "library_version")}
- Horizon: {model.horizon}
- Owner: {model.owner or "not specified"}
- Purpose: {model.purpose or "not specified"}
- Config SHA-256: {_metadata_value(result, "config_sha256")}
- Reference schema SHA-256: {_metadata_value(result, "reference_schema_sha256")}
- Current schema SHA-256: {_metadata_value(result, "current_schema_sha256")}

## Validation Status

- Overall status: {result.status.value}
- Created at: {result.created_at}

## Metrics

{metrics}

## Intended Use

This model card documents quantitative validation evidence for a binary PD model.

## Limitations

This library supports validation evidence and documentation. It does not approve
models, certify regulatory compliance, calculate regulatory capital, or calculate
official provisions.
"""


def _metadata_value(result: "PDValidationResult", key: str) -> str:
    value = result.metadata.get(key)
    return "not available" if value is None else str(value)
