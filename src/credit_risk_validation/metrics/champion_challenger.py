"""Champion versus challenger comparison helpers."""

import polars as pl

from credit_risk_validation.metrics.calibration import calibration_metrics
from credit_risk_validation.metrics.discrimination import auc_gini_ks


def compare_champion_challenger(
    y_true: list[int],
    champion_pd: list[float],
    challenger_pd: list[float],
) -> pl.DataFrame:
    """Compara metricas globales lado a lado sin recomendar reemplazo automatico."""

    champion_discrimination = auc_gini_ks(y_true, champion_pd)
    challenger_discrimination = auc_gini_ks(y_true, challenger_pd)
    champion_calibration, _ = calibration_metrics(y_true, champion_pd, n_bins=10)
    challenger_calibration, _ = calibration_metrics(y_true, challenger_pd, n_bins=10)
    rows = []
    for metric_name in ["auc", "gini", "ks", "brier", "log_loss", "ece", "oe_ratio"]:
        champion_value = (champion_discrimination | champion_calibration)[metric_name].value
        challenger_value = (challenger_discrimination | challenger_calibration)[metric_name].value
        delta = (
            challenger_value - champion_value
            if champion_value is not None and challenger_value is not None
            else None
        )
        rows.append(
            {
                "metric": metric_name,
                "champion": champion_value,
                "challenger": challenger_value,
                "absolute_delta": delta,
                "relative_delta": delta / champion_value
                if delta is not None and champion_value is not None and champion_value != 0
                else None,
                "message": "Review required before any champion replacement decision.",
            }
        )
    return pl.DataFrame(rows)
