"""Download configured Kaggle datasets."""

import argparse
from pathlib import Path

from credit_risk_validation.datasets.kaggle import download_kaggle_resource

DATASET_KEYS = ["give_me_some_credit", "home_credit_stability", "default_credit_card_clients"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=[*DATASET_KEYS, "home_credit_model_stability", "all"], default="all"
    )
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    selected = DATASET_KEYS if args.dataset == "all" else [_normalize(args.dataset)]
    for key in selected:
        try:
            print(f"Downloading {key}")
            download_kaggle_resource(key, output_dir=Path(args.output_dir) / key, unzip=True)
        except Exception as exc:
            print(f"Skipped {key}: {exc}")


def _normalize(dataset: str) -> str:
    if dataset == "home_credit_model_stability":
        return "home_credit_stability"
    return dataset


if __name__ == "__main__":
    main()
