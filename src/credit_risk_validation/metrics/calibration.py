"""Calibration metrics for binary PD models."""

from dataclasses import dataclass
from warnings import catch_warnings, filterwarnings, simplefilter

import numpy as np
import polars as pl
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from credit_risk_validation.config import ThresholdConfig, ValidationOptions
from credit_risk_validation.constants import EPSILON
from credit_risk_validation.schemas import MetricResult
from credit_risk_validation.status import Status, worst_status
from credit_risk_validation.utils.binning import assign_bins, quantile_edges


def calibration_metrics(
    y_true: list[int],
    y_pred_pd: list[float],
    *,
    n_bins: int,
    sample_weight: list[float] | None = None,
    calibration_abs_error_threshold: ThresholdConfig | None = None,
    oe_ratio_threshold: ThresholdConfig | None = None,
) -> tuple[dict[str, MetricResult], pl.DataFrame]:
    """Calcula Brier, Log Loss, ECE, MCE, O/E y tabla de calibracion."""

    context = _metric_context(y_true)
    calibration_threshold = _effective_calibration_threshold(calibration_abs_error_threshold)
    threshold_context = _threshold_context(calibration_threshold)
    oe_threshold_context = _threshold_context(oe_ratio_threshold)
    if len(set(y_true)) < 2:
        message = "Need at least one event and one non-event"
        empty = pl.DataFrame()
        return {
            "brier": _metric_result("brier", None, Status.INSUFFICIENT_DATA, message, context),
            "log_loss": _metric_result(
                "log_loss", None, Status.INSUFFICIENT_DATA, message, context
            ),
            "ece": _metric_result(
                "ece",
                None,
                Status.INSUFFICIENT_DATA,
                message,
                context,
                threshold_context,
            ),
            "mce": _metric_result(
                "mce",
                None,
                Status.INSUFFICIENT_DATA,
                message,
                context,
                threshold_context,
            ),
            "oe_ratio": _metric_result(
                "oe_ratio",
                None,
                Status.INSUFFICIENT_DATA,
                message,
                context,
                oe_threshold_context,
            ),
            "calibration_in_the_large": _metric_result(
                "calibration_in_the_large", None, Status.INSUFFICIENT_DATA, message, context
            ),
            "calibration_intercept": _metric_result(
                "calibration_intercept", None, Status.INSUFFICIENT_DATA, message, context
            ),
            "calibration_slope": _metric_result(
                "calibration_slope", None, Status.INSUFFICIENT_DATA, message, context
            ),
        }, empty

    target = np.asarray(y_true, dtype=int)
    pd_values = np.clip(np.asarray(y_pred_pd, dtype=float), EPSILON, 1 - EPSILON)
    weight = (
        np.ones(len(target), dtype=float) if sample_weight is None else np.asarray(sample_weight)
    )
    table = calibration_table(
        target,
        pd_values,
        weight=weight,
        n_bins=n_bins,
        calibration_abs_error_threshold=calibration_threshold,
    )
    abs_error = np.abs(table["observed_rate"].to_numpy() - table["mean_pd"].to_numpy())
    bin_weight_share = table["weighted_count"].to_numpy() / float(np.sum(weight))
    ece = float(np.sum(abs_error * bin_weight_share))
    mce = float(np.max(abs_error)) if len(abs_error) else 0.0
    observed_events = float(np.sum(target * weight))
    expected_events = float(np.sum(pd_values * weight))
    oe_ratio = observed_events / expected_events if expected_events > 0 else None
    ece_status, ece_message = _upper_threshold_status(ece, calibration_threshold, metric_name="ECE")
    mce_status, mce_message = _upper_threshold_status(mce, calibration_threshold, metric_name="MCE")
    oe_status, oe_message = _two_sided_threshold_status(
        oe_ratio, oe_ratio_threshold, metric_name="O/E ratio"
    )
    calibration_fit = _calibration_intercept_and_slope(target, pd_values, weight)
    return {
        "brier": _metric_result(
            "brier",
            float(brier_score_loss(target, pd_values, sample_weight=weight)),
            Status.OK,
            "",
            context,
        ),
        "log_loss": _metric_result(
            "log_loss",
            float(log_loss(target, pd_values, sample_weight=weight)),
            Status.OK,
            "",
            context,
        ),
        "ece": _metric_result("ece", ece, ece_status, ece_message, context, threshold_context),
        "mce": _metric_result("mce", mce, mce_status, mce_message, context, threshold_context),
        "oe_ratio": _metric_result(
            "oe_ratio",
            oe_ratio,
            oe_status,
            oe_message,
            context,
            oe_threshold_context,
        ),
        "calibration_in_the_large": _metric_result(
            "calibration_in_the_large",
            float(np.average(target - pd_values, weights=weight)),
            Status.OK,
            "",
            context,
        ),
        "calibration_intercept": _metric_result(
            "calibration_intercept",
            calibration_fit.intercept,
            calibration_fit.status,
            calibration_fit.message,
            context,
        ),
        "calibration_slope": _metric_result(
            "calibration_slope",
            calibration_fit.slope,
            calibration_fit.status,
            calibration_fit.message,
            context,
        ),
    }, table


def calibration_table(
    y_true: np.ndarray,
    y_pred_pd: np.ndarray,
    *,
    weight: np.ndarray,
    n_bins: int,
    calibration_abs_error_threshold: ThresholdConfig | None = None,
) -> pl.DataFrame:
    """Genera tabla agregada de calibracion por bins de PD."""

    edges = quantile_edges(y_pred_pd.tolist(), n_bins)
    bins = assign_bins(y_pred_pd.tolist(), edges)
    rows: list[dict[str, object]] = []
    for bin_id in sorted(set(bins.tolist())):
        mask = bins == bin_id
        bin_weight = float(np.sum(weight[mask]))
        events = float(np.sum(y_true[mask] * weight[mask]))
        expected_defaults = float(np.sum(y_pred_pd[mask] * weight[mask]))
        mean_pd = float(np.average(y_pred_pd[mask], weights=weight[mask])) if bin_weight else 0.0
        observed_rate = events / bin_weight if bin_weight else 0.0
        abs_error = abs(observed_rate - mean_pd)
        rel_error = abs_error / mean_pd if mean_pd > 0 else None
        lower_event_rate, upper_event_rate = _wilson_interval(events, bin_weight)
        oe_ratio = events / expected_defaults if expected_defaults > 0 else None
        non_events = bin_weight - events
        status, message = _calibration_bin_status(
            events,
            non_events,
            abs_error,
            calibration_abs_error_threshold,
        )
        rows.append(
            {
                "bin": int(bin_id),
                "n": int(np.sum(mask)),
                "count": int(np.sum(mask)),
                "weighted_count": bin_weight,
                "lower_bound": float(np.min(y_pred_pd[mask])),
                "upper_bound": float(np.max(y_pred_pd[mask])),
                "events": events,
                "non_events": non_events,
                "observed_defaults": events,
                "expected_defaults": expected_defaults,
                "event_rate": observed_rate,
                "avg_pd": mean_pd,
                "mean_pd": mean_pd,
                "observed_rate": observed_rate,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "oe_ratio": oe_ratio,
                "lower_event_rate": lower_event_rate,
                "upper_event_rate": upper_event_rate,
                "status": status.value,
                "message": message,
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
    calibration_abs_error_threshold: ThresholdConfig | None = None,
    oe_ratio_threshold: ThresholdConfig | None = None,
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
        calibration_abs_error_threshold=calibration_abs_error_threshold,
        oe_ratio_threshold=oe_ratio_threshold,
    )


def _calibration_bin_status(
    events: float,
    non_events: float,
    abs_error: float,
    threshold: ThresholdConfig | None,
) -> tuple[Status, str]:
    findings: list[tuple[Status, str]] = []
    if events <= 0:
        findings.append((Status.WARNING, "Calibration bin has no events"))
    if non_events <= 0:
        findings.append((Status.WARNING, "Calibration bin has no non-events"))
    error_status, error_message = _upper_threshold_status(
        abs_error,
        _effective_calibration_threshold(threshold),
        metric_name="Calibration absolute error",
    )
    if error_status != Status.OK:
        findings.append((error_status, error_message))
    if not findings:
        return Status.OK, ""
    return worst_status([status for status, _ in findings]), "; ".join(
        message for _, message in findings
    )


def _wilson_interval(
    events: float, total: float, z: float = 1.96
) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    rate = events / total
    denominator = 1 + z**2 / total
    centre = rate + z**2 / (2 * total)
    margin = z * np.sqrt((rate * (1 - rate) + z**2 / (4 * total)) / total)
    return max(0.0, float((centre - margin) / denominator)), min(
        1.0, float((centre + margin) / denominator)
    )


@dataclass(frozen=True)
class _CalibrationFit:
    intercept: float | None
    slope: float | None
    status: Status
    message: str


def _calibration_intercept_and_slope(
    y_true: np.ndarray, y_pred_pd: np.ndarray, weight: np.ndarray
) -> _CalibrationFit:
    x_values = logit(np.clip(y_pred_pd, EPSILON, 1 - EPSILON))
    active = weight > 0
    active_x = x_values[active]
    active_y = y_true[active]
    if len(active_x) == 0 or len(np.unique(active_y)) < 2:
        return _CalibrationFit(
            None,
            None,
            Status.INSUFFICIENT_DATA,
            "Need positive weight for at least one event and one non-event",
        )
    if len(np.unique(active_x)) < 2:
        return _CalibrationFit(
            None,
            None,
            Status.INSUFFICIENT_DATA,
            "Calibration intercept and slope are not identifiable for constant PD",
        )

    event_x = active_x[active_y == 1]
    non_event_x = active_x[active_y == 0]
    if np.min(event_x) >= np.max(non_event_x) or np.min(non_event_x) >= np.max(event_x):
        return _CalibrationFit(
            None,
            None,
            Status.INSUFFICIENT_DATA,
            "Calibration intercept and slope are not finite under complete or quasi separation",
        )

    try:
        model = LogisticRegression(C=np.inf, fit_intercept=True, solver="lbfgs", max_iter=1000)
        with catch_warnings():
            filterwarnings(
                "ignore",
                message="Setting penalty=None will ignore the C and l1_ratio parameters",
                category=UserWarning,
                module="sklearn.linear_model._logistic",
            )
            simplefilter("error", ConvergenceWarning)
            model.fit(x_values.reshape(-1, 1), y_true, sample_weight=weight)
        intercept = float(np.ravel(model.intercept_)[0])
        slope = float(np.ravel(model.coef_)[0])
        if not np.isfinite(intercept) or not np.isfinite(slope):
            raise ArithmeticError("calibration fit produced non-finite coefficients")
        return _CalibrationFit(intercept, slope, Status.OK, "")
    except ConvergenceWarning:
        return _CalibrationFit(
            None,
            None,
            Status.INSUFFICIENT_DATA,
            "Calibration intercept and slope did not converge",
        )
    except Exception as exc:
        return _CalibrationFit(
            None,
            None,
            Status.ERROR,
            f"Calibration intercept and slope failed: {type(exc).__name__}",
        )


def _metric_context(y_true: list[int]) -> dict[str, int]:
    event_count = int(sum(y_true))
    sample_size = len(y_true)
    return {
        "sample_size": sample_size,
        "event_count": event_count,
        "non_event_count": sample_size - event_count,
    }


def _effective_calibration_threshold(threshold: ThresholdConfig | None) -> ThresholdConfig:
    return threshold or ThresholdConfig(warning=0.02, critical=0.05)


def _upper_threshold_status(
    value: float,
    threshold: ThresholdConfig,
    *,
    metric_name: str,
) -> tuple[Status, str]:
    if threshold.critical is not None and value >= threshold.critical:
        return Status.CRITICAL, f"{metric_name} is at or above critical threshold"
    if threshold.warning is not None and value >= threshold.warning:
        return Status.WARNING, f"{metric_name} is at or above warning threshold"
    return Status.OK, ""


def _two_sided_threshold_status(
    value: float | None,
    threshold: ThresholdConfig | None,
    *,
    metric_name: str,
) -> tuple[Status, str]:
    if value is None or threshold is None:
        return Status.OK, ""
    if threshold.critical_low is not None and value <= threshold.critical_low:
        return Status.CRITICAL, f"{metric_name} is at or below critical lower threshold"
    if threshold.critical_high is not None and value >= threshold.critical_high:
        return Status.CRITICAL, f"{metric_name} is at or above critical upper threshold"
    if threshold.warning_low is not None and value <= threshold.warning_low:
        return Status.WARNING, f"{metric_name} is at or below warning lower threshold"
    if threshold.warning_high is not None and value >= threshold.warning_high:
        return Status.WARNING, f"{metric_name} is at or above warning upper threshold"
    return Status.OK, ""


def _threshold_context(threshold: ThresholdConfig | None) -> dict[str, float | None]:
    return {
        "threshold_warning": threshold.warning if threshold else None,
        "threshold_critical": threshold.critical if threshold else None,
        "threshold_warning_low": threshold.warning_low if threshold else None,
        "threshold_warning_high": threshold.warning_high if threshold else None,
        "threshold_critical_low": threshold.critical_low if threshold else None,
        "threshold_critical_high": threshold.critical_high if threshold else None,
    }


def _metric_result(
    name: str,
    value: float | None,
    status: Status,
    message: str,
    context: dict[str, int],
    threshold_context: dict[str, float | None] | None = None,
) -> MetricResult:
    thresholds = threshold_context or {
        "threshold_warning": None,
        "threshold_critical": None,
        "threshold_warning_low": None,
        "threshold_warning_high": None,
        "threshold_critical_low": None,
        "threshold_critical_high": None,
    }
    return MetricResult(
        name,
        value,
        status,
        message,
        threshold_warning=thresholds["threshold_warning"],
        threshold_critical=thresholds["threshold_critical"],
        threshold_warning_low=thresholds["threshold_warning_low"],
        threshold_warning_high=thresholds["threshold_warning_high"],
        threshold_critical_low=thresholds["threshold_critical_low"],
        threshold_critical_high=thresholds["threshold_critical_high"],
        sample_size=context["sample_size"],
        event_count=context["event_count"],
        non_event_count=context["non_event_count"],
    )
