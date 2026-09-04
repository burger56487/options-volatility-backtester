"""Raw SVI total-variance parameterisation."""

from __future__ import annotations

import numpy as np


def svi_total_variance(
    log_moneyness: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Return w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))."""
    k = np.asarray(log_moneyness, dtype=float)
    return a + b * (
        rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2)
    )


def validate_svi_parameters(
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> None:
    if b <= 0:
        raise ValueError("SVI b must be positive.")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("SVI rho must lie in [-1, 1].")
    if sigma <= 0:
        raise ValueError("SVI sigma must be positive.")
