# Metrics

## Discrimination

AUC measures ranking quality. Gini is `2 * AUC - 1`. KS measures maximum
separation between cumulative event and non-event distributions. KS is evaluated
only at distinct score thresholds, and lift bins never split equal scores.

## Calibration

Brier and Log Loss evaluate probability quality. Calibration bins compare mean
predicted PD with observed default rate. ECE aggregates absolute calibration
error by bin weight. ECE and MCE use upper warning/critical thresholds; O/E uses
two-sided lower and upper thresholds. Calibration intercept and slope are reported
as insufficient when the fit is not identifiable instead of returning penalized
finite coefficients.

## Stability

PSI compares distribution shares between reference and current samples for PD,
score, and configured segment variables. CSI uses the same aggregate
distribution drift calculation for configured non-score variables. Segment drift
tables also compare population share, bad rate, and average PD across reference
and current samples.

## Metric Output Schema

Every metric exported to JSON and metric tables includes audit fields:
`name`, `value`, `reference_value`, `current_value`, `delta`,
`threshold_warning`, `threshold_critical`, `status`, `message`, `sample_size`,
`event_count`, and `non_event_count`. Fields that do not apply to a specific
metric are exported as null rather than omitted.

Metrics with two-sided limits also expose `threshold_warning_low`,
`threshold_warning_high`, `threshold_critical_low`, and
`threshold_critical_high`.
