"""Dataset metadata helpers."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_dataset_metadata(
    dataset_dir: Path,
    *,
    dataset_key: str,
    source: str,
    target_col: str,
    reference_rows: int,
    current_rows: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write reproducibility metadata for a processed dataset."""

    metadata = {
        "dataset_key": dataset_key,
        "source": source,
        "target_col": target_col,
        "reference_rows": reference_rows,
        "current_rows": current_rows,
        "created_at": datetime.now(UTC).isoformat(),
        "notes": "Raw Kaggle data is not committed. Processed files are ignored by git.",
        **(extra or {}),
    }
    path = dataset_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path
