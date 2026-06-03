"""Generate a small HTML report from synthetic data."""

from pathlib import Path

import polars as pl

from credit_risk_validation import PDValidationSuite

frame = pl.DataFrame(
    {
        "target": [0, 1, 0, 0, 1] * 40,
        "pd": [0.03, 0.42, 0.08, 0.12, 0.51] * 40,
        "score": [810, 590, 760, 730, 540] * 40,
    }
)

result = PDValidationSuite(
    target_col="target", pd_col="pd", score_col="score", score_direction="lower_is_riskier"
).run(validation_data=frame)

Path("reports").mkdir(exist_ok=True)
result.to_html("reports/example_report.html")
