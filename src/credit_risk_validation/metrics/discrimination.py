"""Discrimination metrics for binary PD models."""

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from credit_risk_validation.config import ValidationOptions
from credit_risk_validation.schemas import MetricResult
from credit_risk_validation.status import Status


def auc_gini_ks(
    y_true: list[int],
    risk_score: list[float],
    *,
    sample_weight: list[float] | None = None,
) -> dict[str, MetricResult]:
    """Calcula AUC, Gini y KS."""

    context = _metric_context(y_true)
    if len(set(y_true)) < 2:
        message = "Need at least one event and one non-event"
        return {
            "auc": _metric_result("auc", None, Status.INSUFFICIENT_DATA, message, context),
            "gini": _metric_result("gini", None, Status.INSUFFICIENT_DATA, message, context),
            "ks": _metric_result("ks", None, Status.INSUFFICIENT_DATA, message, context),
        }
    auc = float(roc_auc_score(y_true, risk_score, sample_weight=sample_weight))
    gini = 2 * auc - 1
    ks = _ks_statistic(y_true, risk_score, sample_weight)
    return {
        "auc": _metric_result("auc", auc, Status.OK, "", context),
        "gini": _metric_result("gini", gini, Status.OK, "", context),
        "ks": _metric_result("ks", ks, Status.OK, "", context),
    }


def lift_table(
    y_true: list[int],
    risk_score: list[float],
    *,
    n_bins: int,
    sample_weight: list[float] | None = None,
) -> pl.DataFrame:
    """Genera lift y captura acumulada por decil de riesgo."""

    values = np.asarray(risk_score, dtype=float)
    target = np.asarray(y_true, dtype=int)
    weight = (
        np.ones(len(target), dtype=float) if sample_weight is None else np.asarray(sample_weight)
    )
    order = np.argsort(-values)
    bins = np.array_split(order, min(n_bins, len(order)))
    total_events = float(np.sum(target * weight))
    total_weight = float(np.sum(weight))
    overall_rate = total_events / total_weight if total_weight else 0.0

    rows: list[dict[str, object]] = []
    cumulative_events = 0.0
    cumulative_weight = 0.0
    for idx, row_index in enumerate(bins, start=1):
        bin_weight = float(np.sum(weight[row_index]))
        events = float(np.sum(target[row_index] * weight[row_index]))
        non_events = bin_weight - events
        bad_rate = events / bin_weight if bin_weight else 0.0
        cumulative_events += events
        cumulative_weight += bin_weight
        rows.append(
            {
                "bin": idx,
                "count": len(row_index),
                "weighted_count": bin_weight,
                "events": events,
                "non_events": non_events,
                "bad_rate": bad_rate,
                "lift": bad_rate / overall_rate if overall_rate else None,
                "event_capture_rate": events / total_events if total_events else None,
                "cumulative_event_capture": cumulative_events / total_events
                if total_events
                else None,
                "cumulative_population_share": cumulative_weight / total_weight
                if total_weight
                else None,
                "min_score": float(np.min(values[row_index])),
                "max_score": float(np.max(values[row_index])),
            }
        )
    return pl.DataFrame(rows)


def discrimination_from_frame(
    frame: pl.DataFrame,
    *,
    target_col: str,
    score_col: str,
    weight_col: str | None,
    validation: ValidationOptions,
) -> tuple[dict[str, MetricResult], pl.DataFrame]:
    """Calcula metricas de discriminacion desde un DataFrame."""

    y_true = [int(value == validation.positive_class) for value in frame[target_col].to_list()]
    risk_score = [float(value) for value in frame[score_col].to_list()]
    if validation.score_direction in {"lower_is_riskier", "higher_is_safer"}:
        risk_score = [-value for value in risk_score]
    sample_weight = (
        [float(value) for value in frame[weight_col].to_list()]
        if weight_col and weight_col in frame.columns
        else None
    )
    metrics = auc_gini_ks(y_true, risk_score, sample_weight=sample_weight)
    table = lift_table(
        y_true,
        risk_score,
        n_bins=validation.n_bins,
        sample_weight=sample_weight,
    )
    return metrics, table


def _ks_statistic(
    y_true: list[int], risk_score: list[float], sample_weight: list[float] | None = None
) -> float:
    target = np.asarray(y_true, dtype=int)
    score = np.asarray(risk_score, dtype=float)
    weight = (
        np.ones(len(target), dtype=float) if sample_weight is None else np.asarray(sample_weight)
    )
    order = np.argsort(-score)
    target = target[order]
    weight = weight[order]
    event_weight = target * weight
    non_event_weight = (1 - target) * weight
    total_events = float(np.sum(event_weight))
    total_non_events = float(np.sum(non_event_weight))
    if total_events == 0 or total_non_events == 0:
        return float("nan")
    event_cdf = np.cumsum(event_weight) / total_events
    non_event_cdf = np.cumsum(non_event_weight) / total_non_events
    return float(np.max(np.abs(event_cdf - non_event_cdf)))


def _metric_context(y_true: list[int]) -> dict[str, int]:
    event_count = int(sum(y_true))
    sample_size = len(y_true)
    return {
        "sample_size": sample_size,
        "event_count": event_count,
        "non_event_count": sample_size - event_count,
    }


def _metric_result(
    name: str,
    value: float | None,
    status: Status,
    message: str,
    context: dict[str, int],
) -> MetricResult:
    return MetricResult(
        name,
        value,
        status,
        message,
        sample_size=context["sample_size"],
        event_count=context["event_count"],
        non_event_count=context["non_event_count"],
    )
