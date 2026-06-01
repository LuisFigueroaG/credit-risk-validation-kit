"""Hashing helpers for audit metadata."""

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Calcula SHA-256 de un archivo."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
