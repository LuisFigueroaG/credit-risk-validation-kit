"""Calibration metrics for binary PD models."""

import numpy as np
import polars as pl
from scipy.special import logit
from sklearn.metrics import brier_score_loss, log_loss

from credit_risk_validation.config import ValidationOptions
from credit_risk_validation.constants import EPSILON
from credit_risk_validation.schemas import MetricResult
from credit_risk_validation.status import Status
from credit_risk_validation.utils.binning import assign_bins, quantile_edges


def calibration_metrics(
    y_true: list[int],
    y_pred_pd: list[float],
    *,
    n_bins: int,
    sample_weight: list[float] | None = None,
) -> tuple[dict[str, MetricResult], pl.DataFrame]:
    """Calcula Brier, Log Loss, ECE, MCE, O/E y tabla de calibracion."""

    if len(set(y_true)) < 2:
        message = "Need at least one event and one non-event"
        empty = pl.DataFrame()
        return {
            "brier": MetricResult("brier", None, Status.INSUFFICIENT_DATA, message),
            "log_loss": MetricResult("log_loss", None, Status.INSUFFICIENT_DATA, message),
            "ece": MetricResult("ece", None, Status.INSUFFICIENT_DATA, message),
            "mce": MetricResult("mce", None, Status.INSUFFICIENT_DATA, message),
            "oe_ratio": MetricResult("oe_ratio", None, Status.INSUFFICIENT_DATA, message),
        }, empty

    target = np.asarray(y_true, dtype=int)
    pd_values = np.clip(np.asarray(y_pred_pd, dtype=float), EPSILON, 1 - EPSILON)
    weight = (
        np.ones(len(target), dtype=float) if sample_weight is None else np.asarray(sample_weight)
    )
    table = calibration_table(target, pd_values, weight=weight, n_bins=n_bins)
    abs_error = np.abs(table["observed_rate"].to_numpy() - table["mean_pd"].to_numpy())
    bin_weight_share = table["weighted_count"].to_numpy() / float(np.sum(weight))
    ece = float(np.sum(abs_error * bin_weight_share))
    mce = float(np.max(abs_error)) if len(abs_error) else 0.0
    observed_events = float(np.sum(target * weight))
    expected_events = float(np.sum(pd_values * weight))
    oe_ratio = observed_events / expected_events if expected_events > 0 else None
    return {
        "brier": MetricResult(
            "brier", float(brier_score_loss(target, pd_values, sample_weight=weight))
        ),
        "log_loss": MetricResult(
            "log_loss", float(log_loss(target, pd_values, sample_weight=weight))
        ),
        "ece": MetricResult("ece", ece),
        "mce": MetricResult("mce", mce),
        "oe_ratio": MetricResult("oe_ratio", oe_ratio),
        "calibration_in_the_large": MetricResult(
            "calibration_in_the_large", float(np.average(target - pd_values, weights=weight))
        ),
        "calibration_slope": MetricResult(
            "calibration_slope", _calibration_slope(target, pd_values, weight)
        ),
    }, table


def calibration_table(
    y_true: np.ndarray,
    y_pred_pd: np.ndarray,
    *,
    weight: np.ndarray,
    n_bins: int,
) -> pl.DataFrame:
    """Genera tabla agregada de calibracion por bins de PD."""

    edges = quantile_edges(y_pred_pd.tolist(), n_bins)
    bins = assign_bins(y_pred_pd.tolist(), edges)
    rows: list[dict[str, float | int]] = []
    for bin_id in sorted(set(bins.tolist())):
        mask = bins == bin_id
        bin_weight = float(np.sum(weight[mask]))
        events = float(np.sum(y_true[mask] * weight[mask]))
        mean_pd = float(np.average(y_pred_pd[mask], weights=weight[mask])) if bin_weight else 0.0
        observed_rate = events / bin_weight if bin_weight else 0.0
        rows.append(
            {
                "bin": int(bin_id),
                "count": int(np.sum(mask)),
                "weighted_count": bin_weight,
                "events": events,
                "non_events": bin_weight - events,
                "mean_pd": mean_pd,
                "observed_rate": observed_rate,
                "abs_error": abs(observed_rate - mean_pd),
                "min_pd": float(np.min(y_pred_pd[mask])),
                "max_pd": float(np.max(y_pred_pd[mask])),
            }
        )
    return pl.DataFrame(rows)


def calibration_from_frame(
    frame: pl.DataFrame,
    *,
    target_col: str,
    pd_col: str,
    weight_col: str | None,
    validation: ValidationOptions,
) -> tuple[dict[str, MetricResult], pl.DataFrame]:
    """Calcula calibracion desde un DataFrame."""

    y_true = [int(value == validation.positive_class) for value in frame[target_col].to_list()]
    y_pred = [float(value) for value in frame[pd_col].to_list()]
    if validation.clip_pd.enabled:
        y_pred = [
            min(max(value, validation.clip_pd.lower), validation.clip_pd.upper) for value in y_pred
        ]
    sample_weight = (
        [float(value) for value in frame[weight_col].to_list()]
        if weight_col and weight_col in frame.columns
        else None
    )
    return calibration_metrics(
        y_true,
        y_pred,
        n_bins=validation.n_bins,
        sample_weight=sample_weight,
    )


def _calibration_slope(
    y_true: np.ndarray, y_pred_pd: np.ndarray, weight: np.ndarray
) -> float | None:
    try:
        from sklearn.linear_model import LogisticRegression

        x = logit(np.clip(y_pred_pd, EPSILON, 1 - EPSILON)).reshape(-1, 1)
        model = LogisticRegression(fit_intercept=True, solver="lbfgs")
        model.fit(x, y_true, sample_weight=weight)
        return float(model.coef_[0][0])
    except Exception:
        return None
