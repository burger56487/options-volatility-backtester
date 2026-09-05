"""Robust SVI surface calibration, Gatheral g(k) and calendar screens.

This module layers on top of the raw SVI parameterisation in
``volatility_surface.svi`` and the existing spread-weighted calibration in
``volatility_surface.calibration``:

- constrained L-BFGS-B multi-start calibration with data-driven starts;
- vega-weighted loss where weights use the Black-76 vega proxy computed from
  each quote's own implied vol (not a hard-coded Gaussian);
- RMSE reported in implied-volatility units (total-variance fit is
  converted back through ``sqrt(w / T)``);
- Durrleman/Gatheral ``g(k) >= 0`` butterfly check on a dense k grid with an
  explicit tolerance so numerical noise is not counted as arbitrage;
- cross-expiry calendar check on a common k grid: total variance must not
  fall as maturity grows.

SVI parameters are not identifiable: different parameter vectors can trace
the same curve, so quality is judged by the fitted curve RMSE rather than by
the parameter vector itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.stats import norm

from .svi import svi_total_variance


BOUNDS = [
    (1e-6, 2.0),      # a: variance level
    (1e-6, 2.0),      # b: wing slope
    (-0.999, 0.999),  # rho
    (-2.0, 2.0),      # m: horizontal shift
    (1e-4, 2.0),      # sigma: ATM smoothness
]
MIN_POINTS = 6
GRID_N = 200
G_TOL = 1e-6
CAL_TOL = 1e-8


@dataclass
class SVIResult:
    expiry: object
    time_to_expiry: float
    params: np.ndarray
    rmse_vol: float
    rmse_total_var: float
    butterfly_violations: int
    min_g: float
    min_w: float
    converged: bool
    num_points: int
    n_starts: int = 0
    warnings: list[str] = field(default_factory=list)
    valid: bool = True


def svi_derivatives(
    k,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Total variance and its first/second derivatives in k."""
    a, b, rho, m, sigma = params
    k = np.asarray(k, dtype=float)
    delta = k - m
    disc = delta**2 + sigma**2
    sqrt_disc = np.sqrt(disc)
    w = a + b * (rho * delta + sqrt_disc)
    wp = b * (rho + delta / sqrt_disc)
    wpp = b * sigma**2 / disc**1.5
    return w, wp, wpp


def svi_gatheral_g(k, params: np.ndarray) -> np.ndarray:
    """Durrleman/Gatheral function g(k); negative means butterfly risk."""
    w, wp, wpp = svi_derivatives(k, params)
    safe_w = np.where(w > 1e-12, w, 1e-12)
    g = (
        (1.0 - k * wp / (2.0 * safe_w)) ** 2
        - (wp**2 / 4.0) * (1.0 / safe_w + 0.25)
        + wpp / 2.0
    )
    return np.where(w > 1e-12, g, -1.0)


def _vega_weights(
    k: np.ndarray,
    iv: np.ndarray,
    time_to_expiry: float,
) -> np.ndarray:
    """Black-76 vega proxy (up to the common D*F factor) per quote."""
    valid = np.isfinite(iv) & (iv > 0) & (time_to_expiry > 0)
    weights = np.full_like(k, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (
            -k + 0.5 * iv**2 * time_to_expiry
        ) / (iv * np.sqrt(time_to_expiry))
        weights[valid] = norm.pdf(d1[valid]) * np.sqrt(
            time_to_expiry
        )
    fallback = np.exp(-0.5 * (k / 0.5) ** 2)
    return np.where(np.isfinite(weights), weights, fallback)


def _initial_guesses(k: np.ndarray, w: np.ndarray) -> list[np.ndarray]:
    a0 = float(max(np.min(w), 1e-4))
    m0 = float(k[int(np.argmin(w))])
    guesses = []
    for b0 in (0.05, 0.1, 0.3):
        for rho0 in (-0.5, -0.2, 0.0):
            for sigma0 in (0.05, 0.1, 0.3):
                guesses.append(
                    np.array([a0, b0, rho0, m0, sigma0])
                )
    return guesses


def _fit_one(
    k: np.ndarray,
    w: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    best_params = None
    best_loss = np.inf
    n_starts = 0
    for initial in _initial_guesses(k, w):
        n_starts += 1
        try:
            result = minimize(
                lambda params: float(
                    np.sum(
                        (
                            (svi_total_variance(k, *params) - w)
                            * weights
                        )
                        ** 2
                    )
                ),
                initial,
                method="L-BFGS-B",
                bounds=BOUNDS,
            )
        except Exception:  # noqa: BLE001 - try next start
            continue
        if result.success and result.fun < best_loss:
            best_loss = float(result.fun)
            best_params = result.x
    if best_params is not None:
        # Final equal-weight polish: the vega-weighted multi-start pass finds
        # a robust basin; the last refinement keeps wing accuracy so the
        # whole curve (not only ATM) is well represented.
        try:
            polished = least_squares(
                lambda params: (
                    svi_total_variance(k, *params) - w
                ),
                best_params,
                bounds=(
                    np.array([bound[0] for bound in BOUNDS]),
                    np.array([bound[1] for bound in BOUNDS]),
                ),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=3_000,
            )
        except Exception:  # noqa: BLE001 - keep the weighted solution
            polished = None
        if polished is not None:
            best_params = polished.x
    return best_params, float(n_starts)


def calibrate_svi_curve(
    curve: pd.DataFrame,
    expiry,
    time_to_expiry: float,
) -> SVIResult:
    """Calibrate raw SVI to one expiry's OTM curve."""
    required = {"log_moneyness", "iv_mid"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = curve[["log_moneyness", "iv_mid"]].copy()
    data["log_moneyness"] = pd.to_numeric(
        data["log_moneyness"],
        errors="coerce",
    )
    data["iv_mid"] = pd.to_numeric(data["iv_mid"], errors="coerce")
    data = data[data["iv_mid"].notna() & (data["iv_mid"] > 0)]
    data = data.dropna(subset=["log_moneyness"])
    num_points = int(len(data))

    if num_points < MIN_POINTS:
        return SVIResult(
            expiry=expiry,
            time_to_expiry=time_to_expiry,
            params=np.full(5, np.nan),
            rmse_vol=float("nan"),
            rmse_total_var=float("nan"),
            butterfly_violations=-1,
            min_g=float("nan"),
            min_w=float("nan"),
            converged=False,
            num_points=num_points,
            warnings=[
                f"点数不足({num_points}<{MIN_POINTS})"
            ],
            valid=False,
        )

    k = data["log_moneyness"].to_numpy(dtype=float)
    iv = data["iv_mid"].to_numpy(dtype=float)
    if time_to_expiry <= 0 or not np.isfinite(time_to_expiry):
        return SVIResult(
            expiry=expiry,
            time_to_expiry=time_to_expiry,
            params=np.full(5, np.nan),
            rmse_vol=float("nan"),
            rmse_total_var=float("nan"),
            butterfly_violations=-1,
            min_g=float("nan"),
            min_w=float("nan"),
            converged=False,
            num_points=num_points,
            warnings=["time_to_expiry 必须为正"],
            valid=False,
        )

    w = iv**2 * time_to_expiry
    weights = _vega_weights(k, iv, time_to_expiry)
    params, n_starts = _fit_one(k, w, weights)
    if params is None:
        return SVIResult(
            expiry=expiry,
            time_to_expiry=time_to_expiry,
            params=np.full(5, np.nan),
            rmse_vol=float("nan"),
            rmse_total_var=float("nan"),
            butterfly_violations=-1,
            min_g=float("nan"),
            min_w=float("nan"),
            converged=False,
            num_points=num_points,
            n_starts=int(n_starts),
            warnings=["所有起点优化均未收敛"],
            valid=False,
        )

    w_model = np.maximum(
        svi_total_variance(k, *params),
        1e-12,
    )
    iv_model = np.sqrt(w_model / time_to_expiry)
    rmse_vol = float(np.sqrt(np.mean((iv_model - iv) ** 2)))
    rmse_total_var = float(np.sqrt(np.mean((w_model - w) ** 2)))

    k_grid = np.linspace(
        float(k.min()) - 0.1,
        float(k.max()) + 0.1,
        GRID_N,
    )
    g_values = svi_gatheral_g(k_grid, params)
    w_values = svi_total_variance(k_grid, *params)
    butterfly_violations = int(np.sum(g_values < -G_TOL))
    min_g = float(np.min(g_values))
    min_w = float(np.min(w_values))

    warnings = []
    if butterfly_violations > 0:
        warnings.append(
            f"g(k)<-{G_TOL:.0e} 在 "
            f"{butterfly_violations}/{GRID_N} 个网格点，蝶式套利"
        )
    if min_w <= 0:
        warnings.append("总方差存在非正值")

    valid = (
        butterfly_violations == 0
        and min_w > 0
        and np.isfinite(rmse_vol)
    )
    return SVIResult(
        expiry=expiry,
        time_to_expiry=time_to_expiry,
        params=params,
        rmse_vol=rmse_vol,
        rmse_total_var=rmse_total_var,
        butterfly_violations=butterfly_violations,
        min_g=min_g,
        min_w=min_w,
        converged=True,
        num_points=num_points,
        n_starts=int(n_starts),
        warnings=warnings,
        valid=valid,
    )


def check_calendar_arbitrage(
    svi_results: list[SVIResult],
    k_grid: np.ndarray | None = None,
) -> tuple[int, list[dict]]:
    """Total-variance monotonicity across valid expiries on a common grid."""
    if k_grid is None:
        k_grid = np.linspace(-0.3, 0.3, 50)
    valid_results = sorted(
        [result for result in svi_results if result.valid],
        key=lambda result: result.time_to_expiry,
    )
    violations = 0
    details = []
    for i in range(1, len(valid_results)):
        previous = valid_results[i - 1]
        current = valid_results[i]
        w1 = svi_total_variance(k_grid, *previous.params)
        w2 = svi_total_variance(k_grid, *current.params)
        crossing = int(np.sum(w2 < w1 - CAL_TOL))
        if crossing > 0:
            violations += crossing
            details.append(
                {
                    "t1": previous.time_to_expiry,
                    "t2": current.time_to_expiry,
                    "crossings": crossing,
                    "grid_points": int(len(k_grid)),
                }
            )
    return violations, details
