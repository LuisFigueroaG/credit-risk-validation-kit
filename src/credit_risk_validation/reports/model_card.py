"""Markdown model card export."""

from typing import TYPE_CHECKING

from credit_risk_validation.reports.i18n import t, translations

if TYPE_CHECKING:
    from credit_risk_validation.results import PDValidationResult


def render_model_card(result: "PDValidationResult") -> str:
    """Renderiza model card en Markdown."""

    locale = translations(result.config.report.language)
    model = result.config.model
    metrics = "\n".join(
        f"- {metric.name}: {metric.value if metric.value is not None else t(locale, 'not_available')} ({metric.status.value})"
        for metric in result.metrics.values()
    )
    return f"""# {t(locale, "model_card_title")}: {model.name}

## {t(locale, "model_information")}

- {t(locale, "version")}: {model.version}
- {t(locale, "library_version")}: {_metadata_value(result, "library_version", locale)}
- {t(locale, "horizon")}: {model.horizon}
- {t(locale, "owner")}: {model.owner or t(locale, "not_specified")}
- {t(locale, "purpose")}: {model.purpose or t(locale, "not_specified")}
- {t(locale, "config_hash")}: {_metadata_value(result, "config_sha256", locale)}
- {t(locale, "reference_schema_hash")}: {_metadata_value(result, "reference_schema_sha256", locale)}
- {t(locale, "current_schema_hash")}: {_metadata_value(result, "current_schema_sha256", locale)}

## {t(locale, "validation_status")}

- {t(locale, "overall_status")}: {result.status.value}
- {t(locale, "created_at")}: {result.created_at}

## {t(locale, "metrics")}

{metrics}

## {t(locale, "intended_use")}

{t(locale, "intended_use_body")}

## {t(locale, "limitations")}

{t(locale, "disclaimer")}
"""


def _metadata_value(result: "PDValidationResult", key: str, locale: dict[str, str]) -> str:
    value = result.metadata.get(key)
    return t(locale, "not_available") if value is None else str(value)
