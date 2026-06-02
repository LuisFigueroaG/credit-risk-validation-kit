# Datasets

The harness supports:

- Give Me Some Credit
- Default of Credit Card Clients
- Home Credit Stability sample mode

Raw Kaggle data is not committed. If Kaggle requires manual terms acceptance,
accept terms in Kaggle and rerun the download command.

Current environment notes:

- `default_credit_card_clients` downloaded successfully from Kaggle without
  additional terms.
- Direct competition downloads for `give_me_some_credit` and
  `home_credit_stability` returned HTTP 403.
- Public Kaggle dataset mirrors downloaded successfully:
  - `brycecf/give-me-some-credit-dataset`
  - `asyoujie/home-credit-credit-risk-model-stability`
- The harness prepared and validated all three configured datasets with
  `status=OK` in sample mode.
- Prepared examples include simple aggregate segment columns: `age_band`,
  `sex_segment`, and `month_segment`. Home Credit also uses `date_decision` for
  temporal monitoring when available.

For HTTP 403 competition downloads, enter the competition or accept its terms in
the Kaggle UI, then rerun the harness. The helper also tries documented public
mirrors where available.
