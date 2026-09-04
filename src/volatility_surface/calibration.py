"""Per-expiry SVI calibration with spread-weighted least squares."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .svi import svi_total_variance, validate_svi_parameters


@dataclass(frozen=True)
class SviCalibration:
    parameters: dict[str, float]
    cost: float
    n_points: int


def _residuals(
    params: np.ndarray,
    k: np.ndarray,
    w: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    a, b, rho, m, sigma = params
    model = svi_total_variance(k, a, b, rho, m, sigma)
    return (model - w) * np.sqrt(weights)


def calibrate_svi(
    log_moneyness,
    total_variance,
    weights=None,
    minimum_points: int = 5,
) -> SviCalibration:
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if weights is None:
        weights = np.ones_like(k)
    else:
        weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(k) & np.isfinite(w) & (w >= 0) & (weights > 0)
    k, w, weights = k[mask], w[mask], weights[mask]
    if k.size < minimum_points:
        raise ValueError(
            f"At least {minimum_points} points are required for SVI."
        )

    best = None
    for rho0 in (-0.6, 0.0, 0.6):
        initial = np.array([np.mean(w), 0.2, rho0, 0.0, 0.2])
        try:
            result = least_squares(
                _residuals,
                initial,
                args=(k, w, weights),
                bounds=(
                    [-10.0, 1e-6, -1.0, -10.0, 1e-6],
                    [10.0, 10.0, 1.0, 10.0, 10.0],
                ),
                max_nfev=2_000,
            )
        except Exception:  # noqa: BLE001
            continue
        if best is None or result.cost < best.cost:
            best = result

    if best is None:
        raise RuntimeError("SVI calibration failed for all initialisations.")
    a, b, rho, m, sigma = best.x
    validate_svi_parameters(a, b, rho, m, sigma)
    return SviCalibration(
        parameters={
            "a": float(a),
            "b": float(b),
            "rho": float(rho),
            "m": float(m),
            "sigma": float(sigma),
        },
        cost=float(best.cost),
        n_points=int(k.size),
    )
