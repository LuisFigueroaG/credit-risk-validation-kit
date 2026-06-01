"""Small validation helpers."""

from collections.abc import Iterable


def ensure_columns_exist(columns: Iterable[str], available: set[str]) -> list[str]:
    """Retorna columnas faltantes."""

    return [column for column in columns if column and column not in available]
