"""Stability and drift metrics."""

import hashlib
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from credit_risk_validation.config import ThresholdConfig
from credit_risk_validation.constants import EPSILON
from credit_risk_validation.schemas import MetricResult
from credit_risk_validation.status import Status
from credit_risk_validation.utils.binning import assign_bins, quantile_edges


def psi_numeric(
    reference: Sequence[float | int | None],
    current: Sequence[float | int | None],
    *,
    n_bins: int,
) -> tuple[float, pl.DataFrame]:
    """Calcula PSI numerico preservando valores faltantes y no finitos.

    Los bordes se obtienen exclusivamente de los valores finitos de referencia.
    El bin cero agrupa nulos, NaN e infinitos, mientras que los denominadores
    conservan el total original de cada muestra.
    """

    reference_finite = _finite_numeric_values(reference)
    edges = quantile_edges(reference_finite, n_bins)
    ref_bins = _numeric_bins(reference, edges)
    cur_bins = _numeric_bins(current, edges)
    rows: list[dict[str, float | int | str]] = []
    psi_value = 0.0
    for bin_id in sorted(set(ref_bins.tolist()) | set(cur_bins.tolist())):
        ref_count = int(np.sum(ref_bins == bin_id))
        cur_count = int(np.sum(cur_bins == bin_id))
        ref_share = ref_count / max(len(reference), 1)
        cur_share = cur_count / max(len(current), 1)
        safe_ref_share = max(ref_share, EPSILON)
        safe_cur_share = max(cur_share, EPSILON)
        contribution = (safe_cur_share - safe_ref_share) * float(
            np.log(safe_cur_share / safe_ref_share)
        )
        psi_value += contribution
        rows.append(
            {
                "bin": int(bin_id),
                "bucket": "MISSING" if bin_id == 0 else f"BIN_{bin_id}",
                "reference_count": ref_count,
                "current_count": cur_count,
                "reference_share": ref_share,
                "current_share": cur_share,
                "psi": contribution,
            }
        )
    return float(psi_value), pl.DataFrame(rows)


def psi_categorical(reference: list[str], current: list[str]) -> tuple[float, pl.DataFrame]:
    """Calcula PSI para variables categoricas."""

    reference_values = [_category_token(value) for value in reference]
    reference_categories = {
        value for value in reference_values if value not in {"MISSING", "OTHER"}
    }
    current_values = [_normalize_current_category(value, reference_categories) for value in current]
    categories = sorted(set(reference_values) | set(current_values))
    psi_value = 0.0
    rows: list[dict[str, float | int | str]] = []
    for category in categories:
        ref_count = reference_values.count(category)
        cur_count = current_values.count(category)
        ref_share = max(ref_count / max(len(reference), 1), EPSILON)
        cur_share = max(cur_count / max(len(current), 1), EPSILON)
        contribution = (cur_share - ref_share) * float(np.log(cur_share / ref_share))
        psi_value += contribution
        rows.append(
            {
                "category": category,
                "reference_count": ref_count,
                "current_count": cur_count,
                "reference_share": ref_share,
                "current_share": cur_share,
                "psi": contribution,
            }
        )
    return float(psi_value), pl.DataFrame(rows)


def psi_status(value: float, threshold: ThresholdConfig) -> Status:
    """Clasifica PSI segun thresholds."""

    if threshold.critical is not None and value >= threshold.critical:
        return Status.CRITICAL
    if threshold.warning is not None and value >= threshold.warning:
        return Status.WARNING
    return Status.OK


def stability_from_frames(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    pd_col: str,
    score_col: str | None,
    target_col: str,
    segment_cols: list[str] | None = None,
    variable_cols: list[str] | None = None,
    positive_class: int = 1,
    n_bins: int,
    psi_threshold: ThresholdConfig,
) -> tuple[dict[str, MetricResult], dict[str, pl.DataFrame]]:
    """Calcula PSI de PD, score y variables configuradas si estan disponibles."""

    metrics: dict[str, MetricResult] = {}
    tables: dict[str, pl.DataFrame] = {}
    context = _metric_context(current, target_col=target_col, positive_class=positive_class)
    threshold_context = _threshold_context(psi_threshold)

    selected_columns = _stable_unique(
        [
            pd_col,
            *([score_col] if score_col else []),
            *(variable_cols or []),
            *(segment_cols or []),
        ]
    )
    available_columns = [
        column
        for column in selected_columns
        if column in reference.columns and column in current.columns
    ]
    metric_names = _psi_metric_names(
        available_columns,
        pd_col=pd_col,
        score_col=score_col,
    )
    variable_rows: list[dict[str, Any]] = []
    csi_rows: list[dict[str, Any]] = []
    for column in available_columns:
        value, table, variable_type = psi_for_column(
            reference, current, column=column, n_bins=n_bins
        )
        status = psi_status(value, psi_threshold)
        metric_name = metric_names[column]
        metrics[metric_name] = _metric_result(
            metric_name,
            value,
            status,
            f"{variable_type.title()} stability for {column}",
            context,
            threshold_context,
        )
        table_name = metric_name
        tables[table_name] = table
        row = {
            "variable": column,
            "variable_type": variable_type,
            "psi": value,
            "status": status.value,
            "message": f"{variable_type.title()} stability for {column}",
        }
        variable_rows.append(row)
        if column not in {pd_col, score_col}:
            csi_rows.append(
                {
                    "variable": column,
                    "variable_type": variable_type,
                    "csi": value,
                    "status": status.value,
                    "message": f"Characteristic stability for {column}",
                }
            )

    if "psi_pd" not in metrics:
        metrics["psi_pd"] = _metric_result(
            "psi_pd",
            None,
            Status.NOT_APPLICABLE,
            "PD column is not available",
            context,
            threshold_context,
        )
    if score_col and "psi_score" not in metrics:
        metrics["psi_score"] = _metric_result(
            "psi_score",
            None,
            Status.NOT_APPLICABLE,
            "Score column is not available",
            context,
            threshold_context,
        )
    elif not score_col:
        metrics["psi_score"] = _metric_result(
            "psi_score",
            None,
            Status.NOT_APPLICABLE,
            "Score column is not configured",
            context,
            threshold_context,
        )

    tables["psi_by_variable"] = pl.DataFrame(variable_rows)
    tables["csi_by_variable"] = pl.DataFrame(csi_rows)
    tables["segment_drift"] = segment_drift_table(
        reference,
        current,
        segment_cols=segment_cols or [],
        target_col=target_col,
        pd_col=pd_col,
        positive_class=positive_class,
    )
    return metrics, tables


def psi_for_column(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    column: str,
    n_bins: int,
) -> tuple[float, pl.DataFrame, str]:
    """Calcula PSI de una columna numerica o categorica."""

    reference_series = reference[column]
    current_series = current[column]
    if reference_series.dtype.is_numeric() and current_series.dtype.is_numeric():
        value, table = psi_numeric(
            reference_series.to_list(),
            current_series.to_list(),
            n_bins=n_bins,
        )
        return value, _with_variable_columns(table, column, "numeric"), "numeric"

    value, table = psi_categorical(
        [_category_token(value) for value in reference_series.to_list()],
        [_category_token(value) for value in current_series.to_list()],
    )
    return value, _with_variable_columns(table, column, "categorical"), "categorical"


def segment_drift_table(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    segment_cols: list[str],
    target_col: str,
    pd_col: str,
    positive_class: int,
) -> pl.DataFrame:
    """Resume drift poblacional, bad rate y PD promedio por segmento configurado."""

    rows: list[dict[str, Any]] = []
    for column in segment_cols:
        if column not in reference.columns or column not in current.columns:
            continue
        reference_values = [_category_token(value) for value in reference[column].to_list()]
        reference_categories = {
            value for value in reference_values if value not in {"MISSING", "OTHER"}
        }
        current_values = [
            _normalize_current_category(value, reference_categories)
            for value in current[column].to_list()
        ]
        categories = sorted(set(reference_values) | set(current_values))
        for category in categories:
            ref_mask = _category_mask(reference[column].to_list(), category, reference_categories)
            cur_mask = [normalized == category for normalized in current_values]
            ref_count = sum(ref_mask)
            cur_count = sum(cur_mask)
            ref_share = max(ref_count / max(reference.height, 1), EPSILON)
            cur_share = max(cur_count / max(current.height, 1), EPSILON)
            rows.append(
                {
                    "segment_column": column,
                    "segment": category,
                    "reference_count": ref_count,
                    "current_count": cur_count,
                    "reference_share": ref_share,
                    "current_share": cur_share,
                    "population_psi": (cur_share - ref_share)
                    * float(np.log(cur_share / ref_share)),
                    "reference_bad_rate": _masked_rate(
                        reference[target_col].to_list(), ref_mask, positive_class
                    ),
                    "current_bad_rate": _masked_rate(
                        current[target_col].to_list(), cur_mask, positive_class
                    ),
                    "reference_avg_pd": _masked_mean(reference[pd_col].to_list(), ref_mask),
                    "current_avg_pd": _masked_mean(current[pd_col].to_list(), cur_mask),
                }
            )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).with_columns(
        (pl.col("current_bad_rate") - pl.col("reference_bad_rate")).alias("bad_rate_delta"),
        (pl.col("current_avg_pd") - pl.col("reference_avg_pd")).alias("avg_pd_delta"),
    )


def _with_variable_columns(table: pl.DataFrame, column: str, variable_type: str) -> pl.DataFrame:
    return table.with_columns(
        pl.lit(column).alias("variable"),
        pl.lit(variable_type).alias("variable_type"),
    ).select(["variable", "variable_type", *table.columns])


def _category_token(value: object) -> str:
    if value is None:
        return "MISSING"
    text = str(value)
    return text if text else "MISSING"


def _normalize_current_category(value: object, reference_categories: set[str]) -> str:
    token = _category_token(value)
    if token == "MISSING" or token in reference_categories:
        return token
    return "OTHER"


def _category_mask(
    values: list[object], category: str, reference_categories: set[str]
) -> list[bool]:
    mask = []
    for value in values:
        token = (
            _normalize_current_category(value, reference_categories)
            if category == "OTHER"
            else _category_token(value)
        )
        mask.append(token == category)
    return mask


def _masked_rate(values: list[object], mask: list[bool], positive_class: int) -> float | None:
    selected = [value for value, keep in zip(values, mask, strict=False) if keep]
    if not selected:
        return None
    return sum(1 for value in selected if value == positive_class) / len(selected)


def _masked_mean(values: list[object], mask: list[bool]) -> float | None:
    selected: list[float] = []
    for value, keep in zip(values, mask, strict=False):
        if not keep or value is None:
            continue
        try:
            numeric = float(str(value))
        except ValueError:
            continue
        if np.isfinite(numeric):
            selected.append(numeric)
    if not selected:
        return None
    return float(np.mean(selected))


def _safe_metric_name(column: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", column.lower()).strip("_")


def _psi_metric_names(columns: list[str], *, pd_col: str, score_col: str | None) -> dict[str, str]:
    """Genera claves PSI unicas, estables y compatibles para las columnas."""

    reserved_names = {"psi_pd", "psi_score"}
    custom_columns_by_name: dict[str, list[str]] = {}
    for column in columns:
        if column == pd_col or (score_col is not None and column == score_col):
            continue
        base_name = f"psi_{_safe_metric_name(column)}"
        custom_columns_by_name.setdefault(base_name, []).append(column)

    metric_names: dict[str, str] = {}
    for column in columns:
        if column == pd_col:
            metric_names[column] = "psi_pd"
            continue
        if score_col is not None and column == score_col:
            metric_names[column] = "psi_score"
            continue

        base_name = f"psi_{_safe_metric_name(column)}"
        has_collision = base_name in reserved_names or len(custom_columns_by_name[base_name]) > 1
        if has_collision:
            digest = hashlib.sha256(column.encode("utf-8")).hexdigest()[:8]
            metric_names[column] = f"{base_name}_{digest}"
        else:
            metric_names[column] = base_name
    return metric_names


def _finite_numeric_values(values: Sequence[float | int | None]) -> list[float]:
    """Retorna los valores numericos finitos de una muestra."""

    return [float(value) for value in values if value is not None and np.isfinite(float(value))]


def _numeric_bins(values: Sequence[float | int | None], edges: np.ndarray) -> np.ndarray:
    """Asigna bins numericos y reserva cero para faltantes o no finitos."""

    bins = np.zeros(len(values), dtype=int)
    finite_indices: list[int] = []
    finite_values: list[float] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            finite_indices.append(index)
            finite_values.append(numeric)
    if finite_values:
        bins[finite_indices] = assign_bins(finite_values, edges)
    return bins


def _stable_unique(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _metric_context(
    frame: pl.DataFrame, *, target_col: str, positive_class: int
) -> dict[str, int | None]:
    if target_col not in frame.columns:
        return {"sample_size": frame.height, "event_count": None, "non_event_count": None}
    event_count = int((frame[target_col] == positive_class).sum())
    return {
        "sample_size": frame.height,
        "event_count": event_count,
        "non_event_count": frame.height - event_count,
    }


def _threshold_context(threshold: ThresholdConfig) -> dict[str, float | None]:
    return {
        "threshold_warning": threshold.warning,
        "threshold_critical": threshold.critical,
    }


def _metric_result(
    name: str,
    value: float | None,
    status: Status,
    message: str,
    context: dict[str, int | None],
    threshold_context: dict[str, float | None],
) -> MetricResult:
    return MetricResult(
        name,
        value,
        status,
        message,
        threshold_warning=threshold_context["threshold_warning"],
        threshold_critical=threshold_context["threshold_critical"],
        sample_size=context["sample_size"],
        event_count=context["event_count"],
        non_event_count=context["non_event_count"],
    )
