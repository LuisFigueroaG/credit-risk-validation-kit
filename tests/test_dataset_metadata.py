import json
from pathlib import Path

from credit_risk_validation.datasets.metadata import write_dataset_metadata


def test_write_dataset_metadata(tmp_path: Path) -> None:
    path = write_dataset_metadata(
        tmp_path,
        dataset_key="example",
        source="datasets/example/example",
        target_col="target",
        reference_rows=10,
        current_rows=5,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset_key"] == "example"
    assert payload["reference_rows"] == 10
    assert payload["current_rows"] == 5
