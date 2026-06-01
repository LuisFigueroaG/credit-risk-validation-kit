"""Stability and drift metrics."""

import numpy as np
import polars as pl

from credit_risk_validation.config import ThresholdConfig
from credit_risk_validation.constants import EPSILON
from credit_risk_validation.schemas import MetricResult
from credit_risk_validation.status import Status
from credit_risk_validation.utils.binning import assign_bins, quantile_edges


def psi_numeric(
    reference: list[float], current: list[float], *, n_bins: int
) -> tuple[float, pl.DataFrame]:
    """Calcula PSI numerico usando bordes de cuantiles del periodo referencia."""

    edges = quantile_edges(reference, n_bins)
    ref_bins = assign_bins(reference, edges)
    cur_bins = assign_bins(current, edges)
    rows: list[dict[str, float | int]] = []
    psi_value = 0.0
    for bin_id in sorted(set(ref_bins.tolist()) | set(cur_bins.tolist())):
        ref_count = int(np.sum(ref_bins == bin_id))
        cur_count = int(np.sum(cur_bins == bin_id))
        ref_share = max(ref_count / max(len(reference), 1), EPSILON)
        cur_share = max(cur_count / max(len(current), 1), EPSILON)
        contribution = (cur_share - ref_share) * float(np.log(cur_share / ref_share))
        psi_value += contribution
        rows.append(
            {
                "bin": int(bin_id),
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

    categories = sorted(set(reference) | set(current))
    psi_value = 0.0
    rows: list[dict[str, float | int | str]] = []
    for category in categories:
        ref_count = reference.count(category)
        cur_count = current.count(category)
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
    return Status.PASS


def stability_from_frames(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    pd_col: str,
    score_col: str | None,
    n_bins: int,
    psi_threshold: ThresholdConfig,
) -> tuple[dict[str, MetricResult], dict[str, pl.DataFrame]]:
    """Calcula PSI de PD y score si esta disponible."""

    metrics: dict[str, MetricResult] = {}
    tables: dict[str, pl.DataFrame] = {}
    pd_value, pd_table = psi_numeric(
        [float(value) for value in reference[pd_col].to_list()],
        [float(value) for value in current[pd_col].to_list()],
        n_bins=n_bins,
    )
    metrics["psi_pd"] = MetricResult("psi_pd", pd_value, psi_status(pd_value, psi_threshold))
    tables["psi_pd"] = pd_table
    if score_col and score_col in reference.columns and score_col in current.columns:
        score_value, score_table = psi_numeric(
            [float(value) for value in reference[score_col].to_list()],
            [float(value) for value in current[score_col].to_list()],
            n_bins=n_bins,
        )
        metrics["psi_score"] = MetricResult(
            "psi_score", score_value, psi_status(score_value, psi_threshold)
        )
        tables["psi_score"] = score_table
    else:
        metrics["psi_score"] = MetricResult(
            "psi_score", None, Status.NOT_APPLICABLE, "Score column is not available"
        )
    return metrics, tables
