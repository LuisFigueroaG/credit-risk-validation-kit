"""Hashing helpers for audit metadata."""

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl


def file_sha256(path: str | Path) -> str:
    """Calcula SHA-256 de un archivo."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    """Calcula SHA-256 de una estructura JSON con orden estable."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def dataframe_schema_sha256(frame: pl.DataFrame) -> str:
    """Calcula fingerprint auditable de esquema sin hashear valores individuales."""

    schema = {
        "height": frame.height,
        "width": frame.width,
        "columns": [{"name": name, "dtype": str(dtype)} for name, dtype in frame.schema.items()],
    }
    return stable_json_sha256(schema)
