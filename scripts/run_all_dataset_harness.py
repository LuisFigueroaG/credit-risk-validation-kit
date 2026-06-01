"""Run validation harness for prepared datasets."""

import argparse

from credit_risk_validation.datasets.baseline_model import run_dataset_harness

DATASET_KEYS = ["give_me_some_credit", "default_credit_card_clients", "home_credit_stability"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=[*DATASET_KEYS, "home_credit_model_stability", "all"], default="all"
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()
    selected = DATASET_KEYS if args.dataset == "all" else [_normalize(args.dataset)]
    for key in selected:
        print(f"{key}: {run_dataset_harness(key, sample_size=args.sample_size)}")


def _normalize(dataset: str) -> str:
    if dataset == "home_credit_model_stability":
        return "home_credit_stability"
    return dataset


if __name__ == "__main__":
    main()
