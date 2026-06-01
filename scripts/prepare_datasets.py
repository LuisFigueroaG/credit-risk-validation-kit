"""Prepare configured datasets if raw files are present."""

import argparse
from pathlib import Path

from credit_risk_validation.datasets.prepare_default_credit_card_clients import (
    prepare_default_credit_card_clients,
)
from credit_risk_validation.datasets.prepare_give_me_some_credit import prepare_give_me_some_credit
from credit_risk_validation.datasets.prepare_home_credit_stability import (
    prepare_home_credit_stability,
)

PREPARE_FUNCS = {
    "give_me_some_credit": prepare_give_me_some_credit,
    "default_credit_card_clients": prepare_default_credit_card_clients,
    "home_credit_stability": prepare_home_credit_stability,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "give_me_some_credit",
            "home_credit_stability",
            "home_credit_model_stability",
            "default_credit_card_clients",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()
    selected = list(PREPARE_FUNCS) if args.dataset == "all" else [_normalize(args.dataset)]
    raw = Path(args.raw_dir)
    processed = Path(args.output_dir)
    for name in selected:
        try:
            print(f"Preparing {name}")
            PREPARE_FUNCS[name](raw, processed, sample_size=args.sample_size)
        except Exception as exc:
            print(f"Skipped {name}: {exc}")


def _normalize(dataset: str) -> str:
    if dataset == "home_credit_model_stability":
        return "home_credit_stability"
    return dataset


if __name__ == "__main__":
    main()
