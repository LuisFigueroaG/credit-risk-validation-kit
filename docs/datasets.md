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
- `give_me_some_credit` returned HTTP 403 during direct competition download.
- `home_credit_stability` returned HTTP 403 during direct competition download.

For HTTP 403 competition downloads, enter the competition or accept its terms in
the Kaggle UI, then rerun the harness.
