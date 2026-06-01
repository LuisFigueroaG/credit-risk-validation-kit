"""Binning helpers used by metrics."""

import numpy as np


def quantile_edges(values: list[float], n_bins: int) -> np.ndarray:
    """Calcula bordes por cuantiles con duplicados removidos."""

    if not values:
        return np.array([], dtype=float)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(np.asarray(values, dtype=float), quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if min_value == max_value:
            return np.array([min_value - 0.5, max_value + 0.5])
        return np.array([min_value, max_value])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_bins(values: list[float], edges: np.ndarray) -> np.ndarray:
    """Asigna indices de bin empezando en 1."""

    if edges.size < 2:
        return np.ones(len(values), dtype=int)
    return np.digitize(np.asarray(values, dtype=float), edges[1:-1], right=True) + 1
