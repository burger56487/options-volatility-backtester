"""Exact GBM terminal simulation under the risk-neutral measure."""

from __future__ import annotations

import math

import numpy as np


def simulate_gbm_terminal(
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    n_paths: int,
    dividend_yield: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Return terminal prices S_T for exact GBM paths."""
    rng = np.random.default_rng(seed)
    drift = (
        risk_free_rate
        - dividend_yield
        - 0.5 * volatility**2
    ) * time_to_expiry
    diffusion = volatility * math.sqrt(time_to_expiry)
    z = rng.standard_normal(n_paths)
    return spot * np.exp(drift + diffusion * z)
