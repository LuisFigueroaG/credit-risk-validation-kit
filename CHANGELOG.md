# Changelog

## 0.1.0 - Unreleased

- Initial package structure.
- Binary PD validation API and CLI.
- Discrimination, calibration, stability, segment, report, and dataset harness
  foundations.
- Apply calibration thresholds to ECE, MCE, and the two-sided O/E ratio, and
  report non-identifiable calibration slope/intercept explicitly.
- Make KS, lift, segment grouping, and PSI robust to tied scores, missing values,
  and colliding labels.
- Compare drift samples with a consistent predictor and expose current-period
  lift and calibration tables.
- Clarify reference/current evidence in HTML reports, harden untrusted output,
  and honor export anonymization consistently across every artifact format.
