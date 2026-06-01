"""Validation result object and exporters."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from credit_risk_validation._version import __version__
from credit_risk_validation.config import PDValidationConfig
from credit_risk_validation.reports.html import render_html_report
from credit_risk_validation.reports.model_card import render_model_card
from credit_risk_validation.schemas import CheckResult, MetricResult
from credit_risk_validation.status import Status, worst_status


@dataclass
class PDValidationResult:
    """Resultado completo de una ejecucion de validacion PD."""

    config: PDValidationConfig
    checks: list[CheckResult]
    metrics: dict[str, MetricResult]
    tables: dict[str, pl.DataFrame]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def status(self) -> Status:
        statuses = [check.status for check in self.checks] + [
            metric.status for metric in self.metrics.values()
        ]
        return worst_status(statuses)

    def to_dict(self) -> dict[str, Any]:
        """Convierte el resultado a un diccionario JSON-safe."""

        return {
            "version": __version__,
            "created_at": self.created_at,
            "status": self.status.value,
            "metadata": self.metadata,
            "config": self.config.to_public_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "tables": {name: table.to_dicts() for name, table in self.tables.items()},
            "disclaimer": (
                "This library supports validation evidence and documentation. It does not approve "
                "models, certify regulatory compliance, calculate regulatory capital, or calculate "
                "official provisions."
            ),
        }

    def to_json(self, path: str | Path) -> None:
        """Exporta metricas, checks y tablas agregadas a JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    def to_html(self, path: str | Path) -> None:
        """Exporta reporte HTML autocontenido."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_html_report(self), encoding="utf-8")

    def to_tables(self, directory: str | Path, *, file_format: str = "csv") -> None:
        """Exporta tablas agregadas en CSV o Parquet."""

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, table in self.tables.items():
            if table.is_empty():
                continue
            if file_format == "parquet":
                table.write_parquet(output_dir / f"{name}.parquet")
            elif file_format == "csv":
                table.write_csv(output_dir / f"{name}.csv")
            else:
                raise ValueError("file_format must be csv or parquet")

    def to_model_card(self, path: str | Path) -> None:
        """Exporta model card en Markdown."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_model_card(self), encoding="utf-8")
