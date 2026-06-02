from pathlib import Path

import polars as pl

from credit_risk_validation.datasets.baseline_model import run_dataset_harness


def test_harness_skips_missing_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert "skipped" in run_dataset_harness("give_me_some_credit")


def test_harness_verifies_artifacts_and_metrics(
    tmp_path: Path, monkeypatch, sample_frame: pl.DataFrame
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_dir = tmp_path / "data" / "processed" / "give_me_some_credit"
    config_dir = tmp_path / "examples" / "configs"
    dataset_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    sample_frame.write_parquet(dataset_dir / "reference.parquet")
    sample_frame.write_parquet(dataset_dir / "current.parquet")
    (config_dir / "give_me_some_credit.yml").write_text(
        """
model:
  name: "Harness Test"
columns:
  target: "target"
  pd: "pd"
  score: "score"
  segments: ["segment"]
validation:
  n_bins: 5
  min_events: 5
  min_non_events: 5
  min_rows: 20
  score_direction: "lower_is_riskier"
""",
        encoding="utf-8",
    )

    result = run_dataset_harness("give_me_some_credit", sample_size=None)

    assert result.endswith("artifacts=ok; metrics=ok")
    reports_dir = tmp_path / "reports" / "give_me_some_credit"
    assert (reports_dir / "pd_validation_report.html").exists()
    assert (reports_dir / "pd_validation_model_card.md").exists()
    assert (reports_dir / "tables" / "psi_by_variable.csv").exists()
    assert (reports_dir / "tables" / "segment_metrics.csv").exists()
