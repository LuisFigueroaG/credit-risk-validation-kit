"""Non-mutating anonymization projection for exported validation results."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from credit_risk_validation.results import PDValidationResult


_DEFAULT_REPORT_TITLE = "PD Model Validation Report"
_CATEGORY_COLUMNS = {"category", "period", "segment"}
_COLUMN_NAME_COLUMNS = {"column", "segment_column", "variable"}
_SEMANTIC_CATEGORY_VALUES = {"MISSING", "OTHER"}


def project_result_for_export(result: PDValidationResult) -> PDValidationResult:
    """Return an anonymized export-only copy without changing ``result``."""

    if not result.config.report.anonymize:
        return result

    config = result.config.model_copy(deep=True)
    column_aliases = _column_aliases(config.columns)
    category_aliases = _category_aliases(result.tables)
    replacements = {
        **category_aliases,
        **column_aliases,
        **_sensitive_text_aliases(config),
    }

    config.columns.target = column_aliases[config.columns.target]
    config.columns.pd = column_aliases[config.columns.pd]
    config.columns.score = _optional_alias(config.columns.score, column_aliases)
    config.columns.period = _optional_alias(config.columns.period, column_aliases)
    config.columns.weight = _optional_alias(config.columns.weight, column_aliases)
    config.columns.id = _optional_alias(config.columns.id, column_aliases)
    config.columns.segments = [column_aliases[column] for column in config.columns.segments]
    config.model.name = "Anonymous PD Model"
    config.model.owner = "REDACTED" if config.model.owner is not None else None
    config.model.purpose = "REDACTED" if config.model.purpose is not None else None
    if config.report.title != _DEFAULT_REPORT_TITLE:
        config.report.title = "Anonymized Validation Report"

    table_key_aliases = _table_key_aliases(result.tables, result.metrics, column_aliases)
    tables = {
        table_key_aliases.get(name, name): _anonymize_table(
            table,
            column_aliases=column_aliases,
            category_aliases=category_aliases,
            replacements=replacements,
        )
        for name, table in result.tables.items()
    }
    metric_key_aliases = _metric_key_aliases(result.metrics, table_key_aliases)
    metrics = {
        metric_key_aliases.get(name, name): replace(
            metric,
            name=metric_key_aliases.get(metric.name, _replace_text(metric.name, replacements)),
            message=_replace_text(metric.message, replacements),
        )
        for name, metric in result.metrics.items()
    }
    checks = [
        replace(
            check,
            name=_replace_text(check.name, replacements),
            message=_replace_text(check.message, replacements),
            value=_replace_nested(check.value, replacements),
        )
        for check in result.checks
    ]

    return result.__class__(
        config=config,
        checks=checks,
        metrics=metrics,
        tables=tables,
        metadata=_replace_nested(result.metadata, replacements),
        created_at=result.created_at,
    )


def _column_aliases(columns: Any) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for source, alias in [
        (columns.target, "target"),
        (columns.pd, "pd"),
        (columns.score, "score"),
        (columns.period, "period"),
        (columns.weight, "weight"),
        (columns.id, "id"),
    ]:
        if source is not None and source not in aliases:
            aliases[source] = alias
    for index, source in enumerate(columns.segments, start=1):
        aliases.setdefault(source, f"segment_{index}")
    return aliases


def _optional_alias(value: str | None, aliases: dict[str, str]) -> str | None:
    return aliases[value] if value is not None else None


def _sensitive_text_aliases(config: Any) -> dict[str, str]:
    aliases = {
        config.model.name: "Anonymous PD Model",
    }
    if config.model.owner is not None:
        aliases[config.model.owner] = "REDACTED"
    if config.model.purpose is not None:
        aliases[config.model.purpose] = "REDACTED"
    if config.report.title != _DEFAULT_REPORT_TITLE:
        aliases[config.report.title] = "Anonymized Validation Report"
    return aliases


def _category_aliases(tables: dict[str, pl.DataFrame]) -> dict[str, str]:
    values: set[str] = set()
    for table in tables.values():
        for column in _CATEGORY_COLUMNS.intersection(table.columns):
            values.update(str(value) for value in table[column].drop_nulls().to_list())
    anonymized_values = sorted(values - _SEMANTIC_CATEGORY_VALUES)
    return {value: f"value_{index:03d}" for index, value in enumerate(anonymized_values, start=1)}


def _table_key_aliases(
    tables: dict[str, pl.DataFrame],
    metrics: dict[str, Any],
    column_aliases: dict[str, str],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for name, table in tables.items():
        if name not in metrics or "variable" not in table.columns or table.is_empty():
            continue
        variable = str(table["variable"][0])
        alias = column_aliases.get(variable)
        if alias is not None:
            aliases[name] = f"psi_{alias}"
    return aliases


def _metric_key_aliases(
    metrics: dict[str, Any], table_key_aliases: dict[str, str]
) -> dict[str, str]:
    return {name: table_key_aliases[name] for name in metrics if name in table_key_aliases}


def _anonymize_table(
    table: pl.DataFrame,
    *,
    column_aliases: dict[str, str],
    category_aliases: dict[str, str],
    replacements: dict[str, str],
) -> pl.DataFrame:
    if table.is_empty() and table.width == 0:
        return table.clone()

    projected = table.clone()
    expressions: list[pl.Expr] = []
    for column in projected.columns:
        if projected.schema[column] != pl.String:
            continue
        if column in _CATEGORY_COLUMNS:
            expressions.append(
                pl.col(column)
                .map_elements(
                    lambda value: category_aliases.get(value, value),
                    return_dtype=pl.String,
                )
                .alias(column)
            )
        elif column in _COLUMN_NAME_COLUMNS:
            expressions.append(
                pl.col(column)
                .map_elements(
                    lambda value: column_aliases.get(value, _replace_text(value, replacements)),
                    return_dtype=pl.String,
                )
                .alias(column)
            )
        else:
            expressions.append(
                pl.col(column)
                .map_elements(
                    lambda value: _replace_text(value, replacements),
                    return_dtype=pl.String,
                )
                .alias(column)
            )
    return projected.with_columns(expressions) if expressions else projected


def _replace_nested(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _replace_text(value, replacements)
    if isinstance(value, dict):
        return {
            _replace_text(str(key), replacements): _replace_nested(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_nested(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_nested(item, replacements) for item in value)
    return value


def _replace_text(value: str, replacements: dict[str, str]) -> str:
    projected = value
    for source in sorted(replacements, key=len, reverse=True):
        if not source:
            continue
        pattern = re.escape(source)
        if source[0].isalnum() and source[-1].isalnum():
            pattern = rf"(?<!\w){pattern}(?!\w)"
        projected = re.sub(pattern, replacements[source], projected)
    return projected
