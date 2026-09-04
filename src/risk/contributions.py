"""Euler risk contributions with identity checks."""

from __future__ import annotations

import numpy as np


def linear_risk_contributions(
    exposures: np.ndarray,
    covariance: np.ndarray,
    confidence_level: float = 0.95,
    z_score: float = 1.645,
) -> dict:
    """Euler contributions for a linear portfolio: sum == portfolio VaR."""
    exposure = np.asarray(exposures, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    portfolio_variance = float(exposure @ covariance @ exposure)
    if portfolio_variance < 0:
        raise ValueError("Covariance matrix must be positive semi-definite.")
    portfolio_var = z_score * np.sqrt(portfolio_variance)
    if portfolio_variance == 0:
        return {
            "contributions": np.zeros_like(exposure),
            "portfolio_var": 0.0,
            "sum_contributions": 0.0,
            "identity_passed": True,
        }
    gradient = covariance @ exposure / np.sqrt(portfolio_variance)
    contributions = exposure * gradient * z_score
    return {
        "contributions": contributions,
        "portfolio_var": portfolio_var,
        "sum_contributions": float(np.sum(contributions)),
        "identity_passed": bool(
            abs(float(np.sum(contributions)) - portfolio_var) < 1e-8
        ),
    }
