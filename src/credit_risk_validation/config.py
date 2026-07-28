"""Configuration models and YAML loading."""

from math import isfinite
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from credit_risk_validation.constants import (
    DEFAULT_MIN_EVENTS,
    DEFAULT_MIN_NON_EVENTS,
    DEFAULT_MIN_ROWS,
    DEFAULT_N_BINS,
)


class ModelMetadata(BaseModel):
    """Metadatos auditables del modelo validado."""

    name: str = "PD Binary Model"
    version: str = "0.1.0"
    horizon: str = "12m"
    owner: str | None = None
    purpose: str | None = None


class ColumnConfig(BaseModel):
    """Mapeo de columnas de entrada."""

    target: str = "target"
    pd: str = "pd"
    score: str | None = "score"
    period: str | None = None
    weight: str | None = None
    id: str | None = None
    segments: list[str] = Field(default_factory=list)


class ClipPDConfig(BaseModel):
    """Configuracion de clipping para probabilidades."""

    enabled: bool = True
    lower: float = 0.000001
    upper: float = 0.999999

    @field_validator("lower", "upper")
    @classmethod
    def _within_unit_interval(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("clip bounds must be in [0, 1]")
        return value


class ValidationOptions(BaseModel):
    """Opciones generales de validacion."""

    positive_class: int = 1
    n_bins: int = DEFAULT_N_BINS
    binning_strategy: str = "quantile"
    min_bin_size: int = DEFAULT_MIN_ROWS
    min_events: int = DEFAULT_MIN_EVENTS
    min_non_events: int = DEFAULT_MIN_NON_EVENTS
    min_rows: int = DEFAULT_MIN_ROWS
    min_segment_size: int = 20
    max_missing_share: float = 0.20
    pd_boundary_warning_share: float = 0.05
    score_direction: str = "higher_is_riskier"
    clip_pd: ClipPDConfig = Field(default_factory=ClipPDConfig)

    @field_validator("n_bins")
    @classmethod
    def _positive_bins(cls, value: int) -> int:
        if value < 2:
            raise ValueError("n_bins must be at least 2")
        return value

    @field_validator("min_segment_size")
    @classmethod
    def _positive_segment_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("min_segment_size must be positive")
        return value

    @field_validator("max_missing_share", "pd_boundary_warning_share")
    @classmethod
    def _share_in_unit_interval(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("share thresholds must be in [0, 1]")
        return value

    @field_validator("score_direction")
    @classmethod
    def _supported_score_direction(cls, value: str) -> str:
        normalized = value.lower()
        supported = {"higher_is_riskier", "lower_is_riskier", "higher_is_safer"}
        if normalized not in supported:
            raise ValueError(
                "score_direction must be one of: higher_is_riskier, lower_is_riskier, higher_is_safer"
            )
        return normalized


class ThresholdConfig(BaseModel):
    """Thresholds de severidad para metricas clave."""

    warning: float | None = None
    critical: float | None = None
    warning_low: float | None = None
    warning_high: float | None = None
    critical_low: float | None = None
    critical_high: float | None = None

    @field_validator(
        "warning",
        "critical",
        "warning_low",
        "warning_high",
        "critical_low",
        "critical_high",
    )
    @classmethod
    def _finite_threshold(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("threshold values must be finite")
        return value

    @model_validator(mode="after")
    def _ordered_thresholds(self) -> "ThresholdConfig":
        if self.warning is not None and self.critical is not None and self.warning > self.critical:
            raise ValueError("warning threshold must not exceed critical threshold")
        if (
            self.critical_low is not None
            and self.warning_low is not None
            and self.critical_low > self.warning_low
        ):
            raise ValueError("critical_low must not exceed warning_low")
        if (
            self.warning_high is not None
            and self.critical_high is not None
            and self.warning_high > self.critical_high
        ):
            raise ValueError("warning_high must not exceed critical_high")
        if (
            self.warning_low is not None
            and self.warning_high is not None
            and self.warning_low >= self.warning_high
        ):
            raise ValueError("warning_low must be below warning_high")
        if (
            self.critical_low is not None
            and self.critical_high is not None
            and self.critical_low >= self.critical_high
        ):
            raise ValueError("critical_low must be below critical_high")
        low_thresholds = [
            value for value in (self.critical_low, self.warning_low) if value is not None
        ]
        high_thresholds = [
            value for value in (self.warning_high, self.critical_high) if value is not None
        ]
        if low_thresholds and high_thresholds and max(low_thresholds) >= min(high_thresholds):
            raise ValueError("lower thresholds must be below upper thresholds")
        return self


class Thresholds(BaseModel):
    """Coleccion de thresholds configurables."""

    auc_drop: ThresholdConfig = Field(
        default_factory=lambda: ThresholdConfig(warning=0.03, critical=0.05)
    )
    gini_drop: ThresholdConfig = Field(
        default_factory=lambda: ThresholdConfig(warning=0.06, critical=0.10)
    )
    ks_drop: ThresholdConfig = Field(
        default_factory=lambda: ThresholdConfig(warning=0.05, critical=0.10)
    )
    psi: ThresholdConfig = Field(
        default_factory=lambda: ThresholdConfig(warning=0.10, critical=0.25)
    )
    calibration_abs_error: ThresholdConfig = Field(
        default_factory=lambda: ThresholdConfig(warning=0.02, critical=0.05)
    )
    oe_ratio: ThresholdConfig = Field(
        default_factory=lambda: ThresholdConfig(
            warning_low=0.80,
            warning_high=1.25,
            critical_low=0.70,
            critical_high=1.43,
        )
    )


class ReportConfig(BaseModel):
    """Opciones de reporte."""

    language: str = "en"
    title: str = "PD Model Validation Report"
    include_charts: bool = True
    include_model_card: bool = True
    include_methodology: bool = True
    anonymize: bool = True

    @field_validator("language")
    @classmethod
    def _supported_language(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"en", "es"}:
            raise ValueError("language must be one of: en, es")
        return normalized


class PDValidationConfig(BaseModel):
    """Configuracion completa para `PDValidationSuite` y CLI."""

    model: ModelMetadata = Field(default_factory=ModelMetadata)
    columns: ColumnConfig = Field(default_factory=ColumnConfig)
    validation: ValidationOptions = Field(default_factory=ValidationOptions)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PDValidationConfig":
        content = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}
        return cls.model_validate(data)

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
