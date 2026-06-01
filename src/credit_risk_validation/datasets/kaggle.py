"""Kaggle download helpers."""

import subprocess
import zipfile
from pathlib import Path

KAGGLE_CLI = "/home/luchopy/.local/bin/kaggle"

DATASET_ALIASES = {
    "give_me_some_credit": ("competition", "GiveMeSomeCredit"),
    "default_credit_card_clients": ("dataset", "uciml/default-of-credit-card-clients-dataset"),
    "home_credit_stability": ("competition", "home-credit-credit-risk-model-stability"),
    "home_credit_model_stability": ("competition", "home-credit-credit-risk-model-stability"),
}


class KaggleDownloadError(RuntimeError):
    """Raised when Kaggle CLI cannot download a resource."""


def download_kaggle_resource(dataset: str, *, output_dir: Path, unzip: bool = True) -> Path:
    """Download a Kaggle resource without embedding credentials."""

    output_dir.mkdir(parents=True, exist_ok=True)
    kind, slug = DATASET_ALIASES.get(dataset, _infer_resource(dataset))
    if kind == "competition":
        command = [KAGGLE_CLI, "competitions", "download", "-c", slug, "-p", str(output_dir)]
    else:
        command = [KAGGLE_CLI, "datasets", "download", "-d", slug, "-p", str(output_dir)]
    if unzip and kind == "dataset":
        command.append("--unzip")
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        message = (process.stderr or process.stdout).strip()
        raise KaggleDownloadError(
            "Kaggle download failed. If Kaggle requires manual terms acceptance, accept terms "
            f"in the Kaggle UI and retry. Detail: {message}"
        )
    if unzip and kind == "competition":
        for archive in output_dir.glob("*.zip"):
            with zipfile.ZipFile(archive) as zip_file:
                zip_file.extractall(output_dir)
    return output_dir


def _infer_resource(dataset: str) -> tuple[str, str]:
    if "/" in dataset:
        return "dataset", dataset
    return "competition", dataset
