# Validation Methodology

A binary PD model estimates probability of default for a defined horizon. CRVK
does not train the model. It validates outputs by checking data contracts,
discrimination, calibration, stability, and segments.

Reference data is the baseline sample. Current data is a recent, OOT, or
production sample used for stability comparisons.

## Core Concepts

Probability of default (`PD`) is a probability in `[0, 1]` linked to a defined
default horizon, for example 12 months. The observed target is binary: `1` for
default and `0` for non-default under the definition used by the institution.

Reference data is the development, validation, test, or OOT baseline used for
comparison. Current data is a later or production-period sample. A calibration
sample is the sample used to evaluate whether predicted PD levels match observed
default rates. A calibration segment is a portfolio, product, geography, risk
grade, or other group where calibration may differ from the global sample.

## Discrimination vs Calibration

Discrimination evaluates ranking. A model with strong discrimination assigns
higher risk scores or PDs to borrowers who default than to borrowers who do not.
AUC summarizes that ranking. Gini is common in credit risk because it is a
linear transformation of AUC: `Gini = 2 * AUC - 1`. KS measures the maximum
distance between cumulative event and non-event distributions.

Calibration evaluates probability levels. A model can rank borrowers well and
still systematically overpredict or underpredict default rates. Brier Score
penalizes squared probability error. Log Loss penalizes confident wrong
probabilities. Calibration-in-the-large measures average observed minus expected
default rate. Calibration intercept measures systematic offset after logit
transformation, and calibration slope measures whether probabilities are too
extreme or too compressed. O/E ratio compares observed defaults with expected
defaults.

## Stability

PSI compares distribution shares between reference and current samples. It is
useful as a monitoring indicator, but it is not a causal explanation and does
not prove model deterioration by itself. PSI can be sensitive to binning, sample
size, missing values, and new categories.

Temporal monitoring is useful because credit portfolios, macro conditions, data
capture processes, and underwriting policies can change over time. CRVK reports
aggregate temporal metrics when a period column is available.

## Segments and Sparse Defaults

Segment analysis helps identify pockets where a model is unstable or poorly
calibrated even when global metrics look acceptable. Small segments with few
defaults are marked `INSUFFICIENT_DATA`; CRVK avoids forcing AUC or calibration
claims where the evidence is weak.

## Documentation Practices

Validation evidence should record model purpose, horizon, target definition,
data period, columns used, configuration, generated artifacts, and known
limitations. Reports should avoid raw customer rows and sensitive identifiers.

## What CRVK Cannot Conclude

CRVK supports validation, monitoring, and documentation. It does not approve
models, certify regulatory compliance, replace expert judgment, calculate
regulatory capital, or calculate official provisions.

## References

- Basel Committee on Banking Supervision,
  [Working Paper No. 14: Studies on the Validation of Internal Rating Systems](https://www.bis.org/publ/bcbs_wp14.htm).
  This is useful background for empirical validation of rating systems, PD
  discrimination, calibration, and benchmarking. CRVK does not implement IRB
  approval workflows.
- European Banking Authority,
  [EBA/GL/2017/16: Guidelines on PD estimation, LGD estimation and treatment of defaulted exposures](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/model-validation/guidelines-pd-estimation-lgd).
  This is regulatory context for risk-parameter estimation and validation; CRVK
  only supports binary PD output validation evidence.
- Federal Reserve and OCC,
  [SR 11-7 / Supervisory Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm),
  and FDIC
  [FIL-22-2017 adoption guidance](https://www.fdic.gov/news/financial-institution-letters/2017/fil17022.html).
  These sources motivate independent validation, governance, documentation, and
  monitoring discipline.
- Chile CMF,
  [RAN Capitulo 21-6](https://www.cmfchile.cl/portal/principal/613/articles-38797_capitulo_216.pdf)
  and
  [Compendio de Normas Contables para Bancos, Capitulo B-1](https://cmfchile.cl/portal/principal/613/w3-propertyvalue-29911.html).
  These are context for Chilean bank risk management and credit-risk provision
  frameworks. CRVK does not calculate official provisions or certify CMF
  compliance.
- scikit-learn documentation for
  [probability calibration](https://sklearn.org/stable/modules/calibration.html),
  [Brier Score](https://sklearn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html),
  [Log Loss](https://sklearn.org/stable/modules/generated/sklearn.metrics.log_loss.html),
  and [ROC AUC](https://sklearn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
- Siddiqi, *Credit Risk Scorecards*.
- Baesens, Rösch and Scheule, *Credit Risk Analytics*.
- Mays, *Handbook of Credit Scoring*.
