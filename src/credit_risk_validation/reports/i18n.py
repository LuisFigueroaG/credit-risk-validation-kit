"""Locale loading helpers for report renderers."""

from functools import lru_cache
from importlib import resources

import yaml


@lru_cache
def translations(language: str) -> dict[str, str]:
    """Load report translations with English fallback."""

    fallback_path = resources.files("credit_risk_validation").joinpath("locales", "en.yml")
    locale_path = resources.files("credit_risk_validation").joinpath("locales", f"{language}.yml")
    fallback = yaml.safe_load(fallback_path.read_text(encoding="utf-8")) or {}
    selected = (
        yaml.safe_load(locale_path.read_text(encoding="utf-8")) if locale_path.is_file() else {}
    ) or {}
    return {**fallback, **selected}


def t(locale: dict[str, str], key: str) -> str:
    """Return a translated string or the key when missing."""

    return locale.get(key, key)
