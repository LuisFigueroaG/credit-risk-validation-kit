# Metrics

## Discrimination

AUC measures ranking quality. Gini is `2 * AUC - 1`. KS measures maximum
separation between cumulative event and non-event distributions.

## Calibration

Brier and Log Loss evaluate probability quality. Calibration bins compare mean
predicted PD with observed default rate. ECE aggregates absolute calibration
error by bin weight.

## Stability

PSI compares distribution shares between reference and current samples for PD,
score, and configured segment variables. CSI uses the same aggregate
distribution drift calculation for configured non-score variables. Segment drift
tables also compare population share, bad rate, and average PD across reference
and current samples.
