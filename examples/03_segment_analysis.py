"""Segment analysis example."""

import polars as pl

from credit_risk_validation import PDValidationSuite

frame = pl.DataFrame(
    {
        "target": [0, 1, 0, 1, 0, 1] * 50,
        "pd": [0.02, 0.31, 0.07, 0.44, 0.09, 0.52] * 50,
        "score": [820, 630, 770, 600, 750, 560] * 50,
        "segment": ["retail", "retail", "sme", "sme", "retail", "sme"] * 50,
    }
)

result = PDValidationSuite(
    target_col="target",
    pd_col="pd",
    score_col="score",
    segment_cols=["segment"],
    score_direction="lower_is_riskier",
).run(reference_data=frame)

print(result.tables["segment_analysis"])
