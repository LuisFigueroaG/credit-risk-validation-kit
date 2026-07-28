# Stability

Stability monitoring compares reference and current distributions. PSI is
reported for PD, score, and configured segment variables. Characteristic
stability is exported for configured non-score variables using the same
population-share drift calculation.

Categorical stability uses reference categories as the baseline. Missing values
are grouped as `MISSING`, and categories that appear only in the current sample
are grouped as `OTHER`.

Numeric PSI also retains null and non-finite observations in a `MISSING` bucket.
Custom PSI metric names are deterministic and cannot overwrite the reserved
`psi_pd` or `psi_score` results.

The exported `segment_drift` table compares reference and current population
share, bad rate, and average PD for each configured segment. Common interpretive
thresholds are configurable and should be aligned with local model risk policy.
