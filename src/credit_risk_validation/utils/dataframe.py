"""DataFrame helpers for Polars and Pandas compatibility."""

from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

FrameLike = pl.DataFrame | pd.DataFrame


def to_polars(data: FrameLike) -> pl.DataFrame:
    """Convierte Pandas o Polars a Polars sin modificar datos de entrada."""

    if isinstance(data, pl.DataFrame):
        return data.clone()
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data)
    raise TypeError("expected a polars.DataFrame or pandas.DataFrame")


def read_table(path: str | Path) -> pl.DataFrame:
    """Lee CSV o Parquet segun extension."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(file_path)
    raise ValueError(f"Unsupported file extension: {suffix}")


def write_table(table: pl.DataFrame, path: str | Path) -> None:
    """Escribe una tabla como CSV o Parquet segun extension."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        table.write_csv(file_path)
        return
    if suffix in {".parquet", ".pq"}:
        table.write_parquet(file_path)
        return
    raise ValueError(f"Unsupported file extension: {suffix}")


def numeric_series(frame: pl.DataFrame, column: str) -> list[float]:
    """Extrae una columna numerica como lista de float."""

    return [float(value) for value in frame[column].to_list()]


def int_series(frame: pl.DataFrame, column: str) -> list[int]:
    """Extrae una columna como lista de int."""

    return [int(value) for value in frame[column].to_list()]


def safe_value(value: Any) -> Any:
    """Convierte valores no serializables comunes a tipos JSON seguros."""

    if hasattr(value, "item"):
        return value.item()
    return value


def optional_float(value: Any) -> float | None:
    """Convierte un valor escalar a float si es posible."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
