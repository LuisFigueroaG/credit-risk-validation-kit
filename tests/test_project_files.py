from pathlib import Path

import pytest
from pydantic import ValidationError

from credit_risk_validation.config import ReportConfig


def test_required_project_files_exist() -> None:
    for path in [
        "LICENSE",
        "NOTICE",
        "README.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "GOVERNANCE.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "THIRD_PARTY_LICENSES.md",
        "pyproject.toml",
    ]:
        assert Path(path).exists(), path


def test_three_dataset_configs_exist() -> None:
    for key in ["give_me_some_credit", "default_credit_card_clients", "home_credit_stability"]:
        assert Path(f"examples/configs/{key}.yml").exists()


def test_report_language_defaults_to_english_and_supports_spanish() -> None:
    assert ReportConfig().language == "en"
    assert ReportConfig(language="es").language == "es"


def test_report_language_rejects_unsupported_values() -> None:
    with pytest.raises(ValidationError):
        ReportConfig(language="fr")
