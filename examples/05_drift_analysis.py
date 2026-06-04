"""Analyze drift between a reference sample and a current sample."""

import polars as pl

from credit_risk_validation import PDValidationSuite

reference = pl.DataFrame(
    {
        "target": [0, 0, 1, 0, 1, 0] * 30,
        "pd": [0.02, 0.05, 0.35, 0.08, 0.55, 0.12] * 30,
        "score": [800, 760, 610, 720, 560, 700] * 30,
        "segment": ["A", "A", "B", "B", "A", "B"] * 30,
    }
)
current = reference.with_columns((pl.col("pd") * 1.1).clip(0.000001, 0.999999).alias("pd"))

suite = PDValidationSuite(
    target_col="target",
    pd_col="pd",
    score_col="score",
    segment_cols=["segment"],
    score_direction="lower_is_riskier",
)
result = suite.run_drift(reference_data=reference, current_data=current)
print(result.status.value)
