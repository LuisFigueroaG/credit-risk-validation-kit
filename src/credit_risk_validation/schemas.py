"""Typed result schemas used by the validation suite."""

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from credit_risk_validation.status import Status


@dataclass(frozen=True)
class CheckResult:
    """Resultado de una validacion de contrato o calidad."""

    name: str
    status: Status
    message: str
    value: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "value": self.value,
        }


@dataclass(frozen=True)
class MetricResult:
    """Resultado de una metrica numerica."""

    name: str
    value: float | None
    status: Status = Status.OK
    message: str = ""
    reference_value: float | None = None
    current_value: float | None = None
    delta: float | None = None
    threshold_warning: float | None = None
    threshold_critical: float | None = None
    sample_size: int | None = None
    event_count: int | None = None
    non_event_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "reference_value": self.reference_value,
            "current_value": self.current_value,
            "delta": self.delta,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical,
            "status": self.status.value,
            "message": self.message,
            "sample_size": self.sample_size,
            "event_count": self.event_count,
            "non_event_count": self.non_event_count,
        }


@dataclass
class TableBundle:
    """Tablas agregadas producidas por la validacion."""

    tables: dict[str, pl.DataFrame] = field(default_factory=dict)

    def add(self, name: str, table: pl.DataFrame) -> None:
        self.tables[name] = table

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {name: table.to_dicts() for name, table in self.tables.items()}
