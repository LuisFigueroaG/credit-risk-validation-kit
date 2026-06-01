"""High-level public validation API."""

from typing import Any

import polars as pl

from credit_risk_validation._version import __version__
from credit_risk_validation.config import (
    ColumnConfig,
    ModelMetadata,
    PDValidationConfig,
    ReportConfig,
    Thresholds,
    ValidationOptions,
)
from credit_risk_validation.data_contracts import validate_contract
from credit_risk_validation.metrics.calibration import calibration_from_frame
from credit_risk_validation.metrics.discrimination import discrimination_from_frame
from credit_risk_validation.metrics.segmentation import segment_analysis
from credit_risk_validation.metrics.stability import stability_from_frames
from credit_risk_validation.results import PDValidationResult
from credit_risk_validation.schemas import MetricResult
from credit_risk_validation.status import Status, worst_status
from credit_risk_validation.utils.dataframe import FrameLike, optional_float, to_polars
from credit_risk_validation.utils.hashing import dataframe_schema_sha256, stable_json_sha256


class PDValidationSuite:
    """Suite de alto nivel para validar modelos PD binarios."""

    def __init__(
        self,
        *,
        target_col: str = "target",
        pd_col: str = "pd",
        score_col: str | None = "score",
        period_col: str | None = None,
        weight_col: str | None = None,
        id_col: str | None = None,
        segment_cols: list[str] | None = None,
        config: PDValidationConfig | None = None,
        **validation_overrides: Any,
    ) -> None:
        if config is None:
            validation = ValidationOptions.model_validate(validation_overrides or {})
            config = PDValidationConfig(
                model=ModelMetadata(),
                columns=ColumnConfig(
                    target=target_col,
                    pd=pd_col,
                    score=score_col,
                    period=period_col,
                    weight=weight_col,
                    id=id_col,
                    segments=segment_cols or [],
                ),
                validation=validation,
                thresholds=Thresholds(),
                report=ReportConfig(),
            )
        self.config = config

    @classmethod
    def from_config(cls, config: PDValidationConfig) -> "PDValidationSuite":
        return cls(config=config)

    def run(
        self, *, reference_data: FrameLike, current_data: FrameLike | None = None
    ) -> PDValidationResult:
        """Ejecuta validacion sobre muestra referencia y opcionalmente muestra actual."""

        reference = self._prepared(to_polars(reference_data))
        current = self._prepared(to_polars(current_data)) if current_data is not None else None
        columns = self.config.columns
        validation = self.config.validation

        checks = validate_contract(reference, columns, validation, name="reference")
        if current is not None:
            checks.extend(validate_contract(current, columns, validation, name="current"))
        critical = any(check.status == Status.CRITICAL for check in checks)

        metrics: dict[str, MetricResult] = {}
        tables: dict[str, pl.DataFrame] = {}
        if critical:
            metrics["validation"] = MetricResult(
                "validation", None, Status.CRITICAL, "Critical data contract failures"
            )
            tables["data_quality"] = self._data_quality_table(reference, current)
            tables["discrimination"] = pl.DataFrame()
            tables["calibration_bins"] = pl.DataFrame()
            tables["stability_summary"] = pl.DataFrame()
            tables["psi_by_variable"] = pl.DataFrame()
            tables["segment_metrics"] = pl.DataFrame()
            tables["temporal_metrics"] = pl.DataFrame()
            return PDValidationResult(
                self.config, checks, metrics, tables, self._metadata(reference, current)
            )

        score_col = (
            columns.score if columns.score and columns.score in reference.columns else columns.pd
        )
        discrimination, lift = discrimination_from_frame(
            reference,
            target_col=columns.target,
            score_col=score_col,
            weight_col=columns.weight,
            validation=validation,
        )
        calibration, calibration_table = calibration_from_frame(
            reference,
            target_col=columns.target,
            pd_col=columns.pd,
            weight_col=columns.weight,
            validation=validation,
            calibration_abs_error_threshold=self.config.thresholds.calibration_abs_error,
        )
        metrics.update(discrimination)
        metrics.update(calibration)
        tables["data_quality"] = self._data_quality_table(reference, current)
        tables["discrimination"] = _metric_table(discrimination)
        tables["lift_table"] = lift
        tables["calibration_bins"] = calibration_table
        segment_table = segment_analysis(reference, columns=columns, validation=validation)
        tables["segment_metrics"] = segment_table
        tables["segment_analysis"] = segment_table

        if current is not None:
            stability_metrics, stability_tables = stability_from_frames(
                reference,
                current,
                pd_col=columns.pd,
                score_col=columns.score,
                n_bins=validation.n_bins,
                psi_threshold=self.config.thresholds.psi,
            )
            metrics.update(stability_metrics)
            tables.update(stability_tables)
            tables["stability_summary"] = _metric_table(stability_metrics)
            tables["psi_by_variable"] = _psi_variable_summary(stability_metrics)
            tables["current_segment_analysis"] = segment_analysis(
                current, columns=columns, validation=validation
            )
        else:
            metrics["psi_pd"] = MetricResult(
                "psi_pd", None, Status.NOT_APPLICABLE, "Current data was not provided"
            )
            metrics["psi_score"] = MetricResult(
                "psi_score", None, Status.NOT_APPLICABLE, "Current data was not provided"
            )
            tables["stability_summary"] = _metric_table(
                {"psi_pd": metrics["psi_pd"], "psi_score": metrics["psi_score"]}
            )
            tables["psi_by_variable"] = pl.DataFrame()

        tables["temporal_metrics"] = self._temporal_metrics(reference, current)

        return PDValidationResult(
            self.config, checks, metrics, tables, self._metadata(reference, current)
        )

    def _prepared(self, frame: pl.DataFrame) -> pl.DataFrame:
        columns = self.config.columns
        validation = self.config.validation
        if columns.pd in frame.columns and validation.clip_pd.enabled:
            frame = frame.with_columns(
                pl.col(columns.pd)
                .clip(validation.clip_pd.lower, validation.clip_pd.upper)
                .alias(columns.pd)
            )
        return frame

    def _metadata(self, reference: pl.DataFrame, current: pl.DataFrame | None) -> dict[str, Any]:
        return {
            "library_version": __version__,
            "reference_rows": reference.height,
            "current_rows": current.height if current is not None else None,
            "config_sha256": stable_json_sha256(self.config.to_public_dict()),
            "reference_schema_sha256": dataframe_schema_sha256(reference),
            "current_schema_sha256": dataframe_schema_sha256(current)
            if current is not None
            else None,
            "privacy": "Only aggregate validation outputs are exported.",
        }

    def _data_quality_table(
        self, reference: pl.DataFrame, current: pl.DataFrame | None
    ) -> pl.DataFrame:
        return pl.DataFrame(
            [
                self._data_quality_row("reference", reference),
                *([self._data_quality_row("current", current)] if current is not None else []),
            ]
        )

    def _data_quality_row(self, name: str, frame: pl.DataFrame) -> dict[str, Any]:
        target_col = self.config.columns.target
        pd_col = self.config.columns.pd
        events = (
            int((frame[target_col] == self.config.validation.positive_class).sum())
            if target_col in frame.columns
            else None
        )
        non_events = frame.height - events if events is not None else None
        return {
            "dataset": name,
            "rows": frame.height,
            "columns": len(frame.columns),
            "events": events,
            "non_events": non_events,
            "bad_rate": events / frame.height if events is not None and frame.height else None,
            "null_cells": int(sum(frame[column].null_count() for column in frame.columns)),
            "pd_min": optional_float(frame[pd_col].min()) if pd_col in frame.columns else None,
            "pd_max": optional_float(frame[pd_col].max()) if pd_col in frame.columns else None,
        }

    def _temporal_metrics(
        self, reference: pl.DataFrame, current: pl.DataFrame | None
    ) -> pl.DataFrame:
        period_col = self.config.columns.period
        if not period_col or period_col not in reference.columns:
            return pl.DataFrame()
        frames = [("reference", reference)]
        if current is not None and period_col in current.columns:
            frames.append(("current", current))
        rows: list[dict[str, Any]] = []
        for dataset_name, frame in frames:
            period_values = sorted(frame[period_col].unique().to_list(), key=str)
            for period_value in period_values:
                period_frame = frame.filter(pl.col(period_col) == period_value)
                rows.append(
                    self._temporal_metrics_row(dataset_name, str(period_value), period_frame)
                )
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows).sort(["dataset", "period"])

    def _temporal_metrics_row(
        self, dataset_name: str, period_value: str, frame: pl.DataFrame
    ) -> dict[str, Any]:
        target_col = self.config.columns.target
        pd_col = self.config.columns.pd
        score_col = (
            self.config.columns.score
            if self.config.columns.score and self.config.columns.score in frame.columns
            else pd_col
        )
        events = int((frame[target_col] == self.config.validation.positive_class).sum())
        non_events = frame.height - events
        discrimination, _ = discrimination_from_frame(
            frame,
            target_col=target_col,
            score_col=score_col,
            weight_col=self.config.columns.weight,
            validation=self.config.validation,
        )
        calibration, _ = calibration_from_frame(
            frame,
            target_col=target_col,
            pd_col=pd_col,
            weight_col=self.config.columns.weight,
            validation=self.config.validation,
            calibration_abs_error_threshold=self.config.thresholds.calibration_abs_error,
        )
        metric_statuses = [
            metric.status for metric in [*discrimination.values(), *calibration.values()]
        ]
        status = worst_status(metric_statuses) if metric_statuses else Status.NOT_APPLICABLE
        return {
            "dataset": dataset_name,
            "period": period_value,
            "rows": frame.height,
            "events": events,
            "non_events": non_events,
            "bad_rate": events / frame.height if frame.height else None,
            "mean_pd": optional_float(frame[pd_col].mean()),
            "mean_score": optional_float(frame[score_col].mean())
            if score_col in frame.columns
            else None,
            "auc": _metric_value(discrimination, "auc"),
            "gini": _metric_value(discrimination, "gini"),
            "ks": _metric_value(discrimination, "ks"),
            "brier": _metric_value(calibration, "brier"),
            "log_loss": _metric_value(calibration, "log_loss"),
            "ece": _metric_value(calibration, "ece"),
            "oe_ratio": _metric_value(calibration, "oe_ratio"),
            "status": status.value,
            "message": _temporal_message(status, events, non_events),
        }


def _metric_table(metrics: dict[str, MetricResult]) -> pl.DataFrame:
    return pl.DataFrame([metric.to_dict() for metric in metrics.values()])


def _metric_value(metrics: dict[str, MetricResult], name: str) -> float | None:
    metric = metrics.get(name)
    return metric.value if metric is not None else None


def _temporal_message(status: Status, events: int, non_events: int) -> str:
    if status == Status.INSUFFICIENT_DATA:
        return "Need at least one event and one non-event for temporal performance metrics"
    return f"events={events}; non_events={non_events}"


def _psi_variable_summary(metrics: dict[str, MetricResult]) -> pl.DataFrame:
    rows = []
    for name, metric in metrics.items():
        if name.startswith("psi_"):
            rows.append(
                {
                    "variable": name.replace("psi_", ""),
                    "psi": metric.value,
                    "status": metric.status.value,
                    "message": metric.message,
                }
            )
    return pl.DataFrame(rows)
