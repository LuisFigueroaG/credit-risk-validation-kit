"""Segment-level validation summaries."""

import polars as pl

from credit_risk_validation.config import ColumnConfig, ValidationOptions
from credit_risk_validation.metrics.calibration import calibration_from_frame
from credit_risk_validation.metrics.discrimination import discrimination_from_frame
from credit_risk_validation.status import Status
from credit_risk_validation.utils.dataframe import optional_float


def segment_analysis(
    frame: pl.DataFrame,
    *,
    columns: ColumnConfig,
    validation: ValidationOptions,
) -> pl.DataFrame:
    """Calcula metricas principales por segmento configurado."""

    if not columns.segments:
        return pl.DataFrame()
    missing = [column for column in columns.segments if column not in frame.columns]
    if missing:
        return pl.DataFrame(
            [
                {
                    "segment": ",".join(missing),
                    "status": Status.NOT_APPLICABLE.value,
                    "message": "missing",
                }
            ]
        )

    working = frame.with_columns(
        pl.concat_str(
            [pl.col(column).cast(pl.Utf8) for column in columns.segments], separator=" | "
        ).alias("__segment_key")
    )
    rows: list[dict[str, object]] = []
    for segment_key in working["__segment_key"].unique().to_list():
        segment_df = working.filter(pl.col("__segment_key") == segment_key)
        target = segment_df[columns.target]
        events = int((target == validation.positive_class).sum())
        non_events = segment_df.height - events
        mean_pd = optional_float(segment_df[columns.pd].mean())
        if events < validation.min_events or non_events < validation.min_non_events:
            rows.append(
                {
                    "segment": segment_key,
                    "count": segment_df.height,
                    "events": events,
                    "non_events": non_events,
                    "status": Status.INSUFFICIENT_DATA.value,
                    "auc": None,
                    "gini": None,
                    "ks": None,
                    "brier": None,
                    "log_loss": None,
                    "observed_rate": events / segment_df.height if segment_df.height else None,
                    "mean_pd": mean_pd,
                }
            )
            continue
        discrimination, _ = discrimination_from_frame(
            segment_df,
            target_col=columns.target,
            score_col=columns.score or columns.pd,
            weight_col=columns.weight,
            validation=validation,
        )
        calibration, _ = calibration_from_frame(
            segment_df,
            target_col=columns.target,
            pd_col=columns.pd,
            weight_col=columns.weight,
            validation=validation,
        )
        rows.append(
            {
                "segment": segment_key,
                "count": segment_df.height,
                "events": events,
                "non_events": non_events,
                "status": Status.OK.value,
                "auc": discrimination["auc"].value,
                "gini": discrimination["gini"].value,
                "ks": discrimination["ks"].value,
                "brier": calibration["brier"].value,
                "log_loss": calibration["log_loss"].value,
                "observed_rate": events / segment_df.height,
                "mean_pd": mean_pd,
            }
        )
    return pl.DataFrame(rows).sort("segment")
