"""Run validation harness for prepared datasets."""

import argparse
from pathlib import Path

from credit_risk_validation.datasets.baseline_model import run_dataset_harness
from credit_risk_validation.datasets.kaggle import download_kaggle_resource
from credit_risk_validation.datasets.prepare_default_credit_card_clients import (
    prepare_default_credit_card_clients,
)
from credit_risk_validation.datasets.prepare_give_me_some_credit import prepare_give_me_some_credit
from credit_risk_validation.datasets.prepare_home_credit_stability import (
    prepare_home_credit_stability,
)

DATASET_KEYS = ["give_me_some_credit", "default_credit_card_clients", "home_credit_stability"]
PREPARE_FUNCS = {
    "give_me_some_credit": prepare_give_me_some_credit,
    "default_credit_card_clients": prepare_default_credit_card_clients,
    "home_credit_stability": prepare_home_credit_stability,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=[*DATASET_KEYS, "home_credit_model_stability", "all"], default="all"
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    selected = DATASET_KEYS if args.dataset == "all" else [_normalize(args.dataset)]
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    for key in selected:
        if args.download:
            print("Downloading configured dataset")
            download_kaggle_resource(key, output_dir=raw_dir / key, unzip=True)
        if args.prepare:
            print("Preparing configured dataset")
            PREPARE_FUNCS[key](
                raw_dir,
                output_dir,
                sample_size=args.sample_size,
                seed=args.seed,
            )
        print(
            f"{key}: {run_dataset_harness(key, sample_size=args.sample_size, processed_dir=output_dir)}"
        )


def _normalize(dataset: str) -> str:
    if dataset == "home_credit_model_stability":
        return "home_credit_stability"
    return dataset


if __name__ == "__main__":
    main()
