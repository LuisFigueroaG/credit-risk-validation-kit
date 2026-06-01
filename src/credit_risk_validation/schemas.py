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
    status: Status = Status.PASS
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "status": self.status.value,
            "message": self.message,
        }


@dataclass
class TableBundle:
    """Tablas agregadas producidas por la validacion."""

    tables: dict[str, pl.DataFrame] = field(default_factory=dict)

    def add(self, name: str, table: pl.DataFrame) -> None:
        self.tables[name] = table

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {name: table.to_dicts() for name, table in self.tables.items()}
