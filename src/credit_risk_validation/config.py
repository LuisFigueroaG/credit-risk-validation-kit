"""Configuration models and YAML loading."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

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
    score_direction: str = "higher_is_riskier"
    clip_pd: ClipPDConfig = Field(default_factory=ClipPDConfig)

    @field_validator("n_bins")
    @classmethod
    def _positive_bins(cls, value: int) -> int:
        if value < 2:
            raise ValueError("n_bins must be at least 2")
        return value


class ThresholdConfig(BaseModel):
    """Thresholds de severidad para metricas clave."""

    warning: float | None = None
    critical: float | None = None
    warning_low: float | None = None
    warning_high: float | None = None
    critical_low: float | None = None
    critical_high: float | None = None


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
