import math

import pytest
from pydantic import ValidationError

from credit_risk_validation.config import ThresholdConfig


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
@pytest.mark.parametrize(
    "field",
    [
        "warning",
        "critical",
        "warning_low",
        "warning_high",
        "critical_low",
        "critical_high",
    ],
)
def test_threshold_values_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(ValidationError, match="threshold values must be finite"):
        ThresholdConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "warning",
        "critical",
        "warning_low",
        "warning_high",
        "critical_low",
        "critical_high",
    ],
)
def test_threshold_values_must_be_non_negative(field: str) -> None:
    with pytest.raises(ValidationError, match="threshold values must be non-negative"):
        ThresholdConfig(**{field: -0.01})


@pytest.mark.parametrize(
    "values",
    [
        {"warning": 0.2, "critical": 0.1},
        {"critical_low": 0.8, "warning_low": 0.7},
        {"warning_high": 1.3, "critical_high": 1.2},
        {"warning_low": 1.2, "warning_high": 1.2},
        {"critical_low": 1.3, "critical_high": 1.2},
        {"critical_low": 1.0, "warning_high": 0.9},
        {"warning_low": 1.0, "critical_high": 0.9},
    ],
)
def test_threshold_values_must_be_ordered(values: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        ThresholdConfig(**values)


def test_valid_two_sided_thresholds_are_accepted() -> None:
    threshold = ThresholdConfig(
        critical_low=0.7,
        warning_low=0.8,
        warning_high=1.25,
        critical_high=1.43,
    )

    assert threshold.warning_low == pytest.approx(0.8)
    assert threshold.warning_high == pytest.approx(1.25)
