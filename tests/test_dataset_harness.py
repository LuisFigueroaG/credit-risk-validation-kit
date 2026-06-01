from pathlib import Path

from credit_risk_validation.datasets.baseline_model import run_dataset_harness


def test_harness_skips_missing_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert "skipped" in run_dataset_harness("give_me_some_credit")
