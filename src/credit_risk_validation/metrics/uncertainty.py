"""Uncertainty helpers."""

from math import sqrt


def wilson_interval(
    successes: float, total: float, z: float = 1.96
) -> tuple[float | None, float | None]:
    """Calcula intervalo Wilson para una proporcion."""

    if total <= 0:
        return None, None
    p_hat = successes / total
    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)
