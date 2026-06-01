from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def sample_frame() -> pl.DataFrame:
    rows = 240
    target = ([0, 0, 0, 1, 0, 1] * (rows // 6))[:rows]
    pd_values = ([0.02, 0.04, 0.08, 0.35, 0.12, 0.55] * (rows // 6))[:rows]
    score = ([820, 780, 740, 620, 700, 560] * (rows // 6))[:rows]
    segment = (["A", "A", "B", "B", "A", "A"] * (rows // 6))[:rows]
    return pl.DataFrame({"target": target, "pd": pd_values, "score": score, "segment": segment})


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(
        """
model:
  name: "Test PD"
columns:
  target: "target"
  pd: "pd"
  score: "score"
  segments: ["segment"]
validation:
  positive_class: 1
  n_bins: 5
  min_events: 5
  min_non_events: 5
  min_rows: 20
  score_direction: "lower_is_riskier"
report:
  title: "Test Report"
""",
        encoding="utf-8",
    )
    return path
