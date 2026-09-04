"""Order-flow models: Hawkes arrivals and informed-flow splits."""

from __future__ import annotations

import numpy as np


def hawkes_intensity_path(
    baseline: float,
    alpha: float,
    beta: float,
    n_steps: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a Hawkes intensity path with self-exciting arrivals."""
    rng = np.random.default_rng(seed)
    intensity = np.empty(n_steps)
    lamb = baseline
    for i in range(n_steps):
        intensity[i] = lamb
        arrival = rng.random() < lamb / (lamb + 1.0)  # scaled probability
        lamb = baseline + alpha * float(arrival) + (1.0 - beta) * (lamb - baseline)
        lamb = max(lamb, 0.0)
    return intensity


def split_informed_flow(
    n_orders: int,
    informed_fraction: float,
    seed: int | None = None,
) -> np.ndarray:
    """Boolean mask marking informed vs uninformed client orders."""
    if not 0.0 <= informed_fraction <= 1.0:
        raise ValueError("informed_fraction must lie in [0, 1].")
    rng = np.random.default_rng(seed)
    return rng.random(n_orders) < informed_fraction
