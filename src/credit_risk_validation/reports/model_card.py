"""Markdown model card export."""

import re
from html import escape
from typing import TYPE_CHECKING

from credit_risk_validation.reports.i18n import t, translations

if TYPE_CHECKING:
    from credit_risk_validation.results import PDValidationResult


MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+.!|>\-])")


def render_model_card(result: "PDValidationResult") -> str:
    """Renderiza model card en Markdown."""

    locale = translations(result.config.report.language)
    model = result.config.model
    metrics = "\n".join(
        f"- {_markdown_text(metric.name)}: "
        f"{metric.value if metric.value is not None else t(locale, 'not_available')} "
        f"({_markdown_text(metric.status.value)})"
        for metric in result.metrics.values()
    )
    return f"""# {t(locale, "model_card_title")}: {_markdown_text(model.name)}

## {t(locale, "model_information")}

- {t(locale, "version")}: {_markdown_text(model.version)}
- {t(locale, "library_version")}: {_metadata_value(result, "library_version", locale)}
- {t(locale, "horizon")}: {_markdown_text(model.horizon)}
- {t(locale, "owner")}: {_markdown_text(model.owner or t(locale, "not_specified"))}
- {t(locale, "purpose")}: {_markdown_text(model.purpose or t(locale, "not_specified"))}
- {t(locale, "config_hash")}: {_metadata_value(result, "config_sha256", locale)}
- {t(locale, "reference_schema_hash")}: {_metadata_value(result, "reference_schema_sha256", locale)}
- {t(locale, "current_schema_hash")}: {_metadata_value(result, "current_schema_sha256", locale)}

## {t(locale, "validation_status")}

- {t(locale, "overall_status")}: {result.status.value}
- {t(locale, "created_at")}: {_markdown_text(result.created_at)}

## {t(locale, "metrics")}

{metrics}

## {t(locale, "intended_use")}

{t(locale, "intended_use_body")}

## {t(locale, "limitations")}

{t(locale, "disclaimer")}
"""


def _metadata_value(result: "PDValidationResult", key: str, locale: dict[str, str]) -> str:
    value = result.metadata.get(key)
    return _markdown_text(t(locale, "not_available") if value is None else str(value))


def _markdown_text(value: object) -> str:
    """Neutraliza HTML y sintaxis Markdown en valores no confiables de una linea."""

    single_line = " ".join(str(value).splitlines())
    html_safe = escape(single_line, quote=True)
    return MARKDOWN_SPECIAL.sub(r"\\\1", html_safe)
