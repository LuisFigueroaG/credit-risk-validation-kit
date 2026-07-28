"""High-level public validation API."""

from dataclasses import replace
from typing import Any

import polars as pl

from credit_risk_validation._version import __version__
from credit_risk_validation.config import (
    ColumnConfig,
    ModelMetadata,
    PDValidationConfig,
    ReportConfig,
    ThresholdConfig,
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
        self,
        *,
        validation_data: FrameLike | None = None,
        reference_data: FrameLike | None = None,
        current_data: FrameLike | None = None,
    ) -> PDValidationResult:
        """Ejecuta validacion principal sobre una muestra evaluada."""

        if validation_data is not None and reference_data is not None:
            raise ValueError("Pass either validation_data or reference_data, not both")
        if current_data is not None:
            drift_reference = validation_data if validation_data is not None else reference_data
            if drift_reference is None:
                raise ValueError("run with current_data requires reference_data or validation_data")
            return self.run_drift(reference_data=drift_reference, current_data=current_data)
        if validation_data is None and reference_data is None:
            raise ValueError("run requires validation_data")

        source_data = validation_data if validation_data is not None else reference_data
        if source_data is None:
            raise ValueError("run requires validation_data")
        validation_frame = self._prepared(to_polars(source_data))
        columns = self.config.columns
        validation = self.config.validation

        checks = validate_contract(validation_frame, columns, validation, name="validation")
        critical = any(check.status == Status.CRITICAL for check in checks)

        metrics: dict[str, MetricResult] = {}
        tables: dict[str, pl.DataFrame] = {}
        if critical:
            metrics["validation"] = MetricResult(
                "validation", None, Status.CRITICAL, "Critical data contract failures"
            )
            tables["data_quality"] = self._data_quality_table(validation_frame, None)
            tables["discrimination"] = pl.DataFrame()
            tables["calibration_bins"] = pl.DataFrame()
            tables["segment_metrics"] = pl.DataFrame()
            tables["segment_analysis"] = pl.DataFrame()
            tables["temporal_metrics"] = pl.DataFrame()
            return PDValidationResult(
                self.config,
                checks,
                metrics,
                tables,
                self._metadata(validation_frame, None, analysis_type="validation"),
            )

        score_col = (
            columns.score
            if columns.score and columns.score in validation_frame.columns
            else columns.pd
        )
        discrimination, lift = discrimination_from_frame(
            validation_frame,
            target_col=columns.target,
            score_col=score_col,
            weight_col=columns.weight,
            validation=validation,
        )
        calibration, calibration_table = calibration_from_frame(
            validation_frame,
            target_col=columns.target,
            pd_col=columns.pd,
            weight_col=columns.weight,
            validation=validation,
            calibration_abs_error_threshold=self.config.thresholds.calibration_abs_error,
            oe_ratio_threshold=self.config.thresholds.oe_ratio,
        )
        metrics.update(discrimination)
        metrics.update(calibration)
        tables["data_quality"] = self._data_quality_table(validation_frame, None)
        tables["discrimination"] = _metric_table(discrimination)
        tables["lift_table"] = lift
        tables["calibration_bins"] = calibration_table
        segment_table = segment_analysis(validation_frame, columns=columns, validation=validation)
        tables["segment_metrics"] = segment_table
        tables["segment_analysis"] = segment_table
        tables["temporal_metrics"] = self._temporal_metrics(validation_frame, None)

        return PDValidationResult(
            self.config,
            checks,
            metrics,
            tables,
            self._metadata(validation_frame, None, analysis_type="validation"),
        )

    def run_drift(
        self, *, reference_data: FrameLike, current_data: FrameLike
    ) -> PDValidationResult:
        """Ejecuta analisis de drift entre muestra referencia y muestra actual."""

        reference = self._prepared(to_polars(reference_data))
        current = self._prepared(to_polars(current_data))
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
            tables["csi_by_variable"] = pl.DataFrame()
            tables["segment_drift"] = pl.DataFrame()
            tables["segment_metrics"] = pl.DataFrame()
            tables["temporal_metrics"] = pl.DataFrame()
            return PDValidationResult(
                self.config,
                checks,
                metrics,
                tables,
                self._metadata(reference, current, analysis_type="drift"),
            )

        effective_columns, effective_validation = _effective_drift_discrimination_config(
            reference,
            current,
            columns=columns,
            validation=validation,
        )
        score_col = effective_columns.score or effective_columns.pd
        discrimination, lift = discrimination_from_frame(
            reference,
            target_col=columns.target,
            score_col=score_col,
            weight_col=columns.weight,
            validation=effective_validation,
        )
        calibration, calibration_table = calibration_from_frame(
            reference,
            target_col=columns.target,
            pd_col=columns.pd,
            weight_col=columns.weight,
            validation=validation,
            calibration_abs_error_threshold=self.config.thresholds.calibration_abs_error,
            oe_ratio_threshold=self.config.thresholds.oe_ratio,
        )
        if current is not None:
            current_discrimination, current_lift = discrimination_from_frame(
                current,
                target_col=columns.target,
                score_col=score_col,
                weight_col=columns.weight,
                validation=effective_validation,
            )
            current_calibration, current_calibration_table = calibration_from_frame(
                current,
                target_col=columns.target,
                pd_col=columns.pd,
                weight_col=columns.weight,
                validation=validation,
                calibration_abs_error_threshold=self.config.thresholds.calibration_abs_error,
                oe_ratio_threshold=self.config.thresholds.oe_ratio,
            )
            discrimination = _attach_current_values(
                discrimination,
                current_discrimination,
                thresholds={
                    "auc": self.config.thresholds.auc_drop,
                    "gini": self.config.thresholds.gini_drop,
                    "ks": self.config.thresholds.ks_drop,
                },
            )
            calibration = _attach_current_values(calibration, current_calibration)
        metrics.update(discrimination)
        metrics.update(calibration)
        tables["data_quality"] = self._data_quality_table(reference, current)
        tables["discrimination"] = _metric_table(discrimination)
        tables["lift_table"] = lift
        tables["calibration_bins"] = calibration_table
        tables["current_lift_table"] = current_lift
        tables["current_calibration_bins"] = current_calibration_table
        segment_table = segment_analysis(
            reference,
            columns=effective_columns,
            validation=effective_validation,
        )
        tables["segment_metrics"] = segment_table
        tables["segment_analysis"] = segment_table

        stability_metrics, stability_tables = stability_from_frames(
            reference,
            current,
            pd_col=columns.pd,
            score_col=columns.score,
            target_col=columns.target,
            segment_cols=columns.segments,
            variable_cols=columns.segments,
            positive_class=validation.positive_class,
            n_bins=validation.n_bins,
            psi_threshold=self.config.thresholds.psi,
        )
        metrics.update(stability_metrics)
        tables.update(stability_tables)
        tables["stability_summary"] = _metric_table(stability_metrics)
        if "psi_by_variable" not in stability_tables:
            tables["psi_by_variable"] = _psi_variable_summary(stability_metrics)
        if "csi_by_variable" not in stability_tables:
            tables["csi_by_variable"] = pl.DataFrame()
        if "segment_drift" not in stability_tables:
            tables["segment_drift"] = pl.DataFrame()
        tables["current_segment_analysis"] = segment_analysis(
            current,
            columns=effective_columns,
            validation=effective_validation,
        )

        tables["temporal_metrics"] = self._temporal_metrics(reference, current)

        return PDValidationResult(
            self.config,
            checks,
            metrics,
            tables,
            self._metadata(reference, current, analysis_type="drift"),
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

    def _metadata(
        self, reference: pl.DataFrame, current: pl.DataFrame | None, *, analysis_type: str
    ) -> dict[str, Any]:
        metadata = {
            "analysis_type": analysis_type,
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
        if analysis_type == "validation":
            metadata["validation_rows"] = reference.height
            metadata["validation_schema_sha256"] = dataframe_schema_sha256(reference)
        return metadata

    def _data_quality_table(
        self, reference: pl.DataFrame, current: pl.DataFrame | None
    ) -> pl.DataFrame:
        first_name = "validation" if current is None else "reference"
        return pl.DataFrame(
            [
                self._data_quality_row(first_name, reference),
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
        frames = [("validation" if current is None else "reference", reference)]
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
            oe_ratio_threshold=self.config.thresholds.oe_ratio,
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


def _effective_drift_discrimination_config(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    columns: ColumnConfig,
    validation: ValidationOptions,
) -> tuple[ColumnConfig, ValidationOptions]:
    configured_score = columns.score
    if (
        configured_score
        and configured_score in reference.columns
        and configured_score in current.columns
    ):
        return columns, validation

    return (
        columns.model_copy(update={"score": columns.pd}),
        validation.model_copy(update={"score_direction": "higher_is_riskier"}),
    )


def _attach_current_values(
    reference_metrics: dict[str, MetricResult],
    current_metrics: dict[str, MetricResult],
    *,
    thresholds: dict[str, ThresholdConfig] | None = None,
) -> dict[str, MetricResult]:
    return {
        name: _metric_with_current(
            metric,
            current_metrics.get(name),
            threshold=(thresholds or {}).get(name),
        )
        for name, metric in reference_metrics.items()
    }


def _metric_with_current(
    reference_metric: MetricResult,
    current_metric: MetricResult | None,
    *,
    threshold: ThresholdConfig | None = None,
) -> MetricResult:
    if current_metric is None:
        return replace(
            reference_metric,
            reference_value=reference_metric.value,
            threshold_warning=threshold.warning
            if threshold
            else reference_metric.threshold_warning,
            threshold_critical=threshold.critical
            if threshold
            else reference_metric.threshold_critical,
        )

    reference_value = reference_metric.value
    current_value = current_metric.value
    delta = (
        current_value - reference_value
        if reference_value is not None and current_value is not None
        else None
    )
    status = worst_status([reference_metric.status, current_metric.status])
    message = reference_metric.message or current_metric.message
    warning_threshold = threshold.warning if threshold else reference_metric.threshold_warning
    critical_threshold = threshold.critical if threshold else reference_metric.threshold_critical

    if (
        threshold is not None
        and reference_value is not None
        and current_value is not None
        and current_metric.status == Status.OK
    ):
        degradation = reference_value - current_value
        if threshold.critical is not None and degradation >= threshold.critical:
            status = Status.CRITICAL
            message = f"{reference_metric.name} degraded by {degradation:.6g}"
        elif threshold.warning is not None and degradation >= threshold.warning:
            status = Status.WARNING
            message = f"{reference_metric.name} degraded by {degradation:.6g}"

    return replace(
        reference_metric,
        status=status,
        message=message,
        reference_value=reference_value,
        current_value=current_value,
        delta=delta,
        threshold_warning=warning_threshold,
        threshold_critical=critical_threshold,
    )


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
