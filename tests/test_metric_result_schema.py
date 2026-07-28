from credit_risk_validation.schemas import MetricResult


def test_metric_result_serializes_two_sided_thresholds() -> None:
    result = MetricResult(
        "oe_ratio",
        1.0,
        threshold_warning_low=0.8,
        threshold_warning_high=1.25,
        threshold_critical_low=0.7,
        threshold_critical_high=1.43,
    )

    serialized = result.to_dict()

    assert serialized["threshold_warning_low"] == 0.8
    assert serialized["threshold_warning_high"] == 1.25
    assert serialized["threshold_critical_low"] == 0.7
    assert serialized["threshold_critical_high"] == 1.43
