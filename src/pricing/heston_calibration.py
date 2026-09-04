"""Heston parameter calibration to European option quotes.

The objective is the RMSE between semi-analytic Heston prices and market
prices. Calibration is multi-start and bounded because the Heston model is
known to be weakly identifiable for some parameter regions; the caller should
always inspect the achieved RMSE and parameter stability across starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.domain.enums import OptionType
from src.pricing.heston import heston_prices_semi_analytic


@dataclass(frozen=True)
class HestonQuote:
    """One European option observation used for calibration."""

    strike: float
    time_to_expiry: float
    option_type: OptionType
    market_price: float


@dataclass(frozen=True)
class HestonCalibrationResult:
    parameters: dict
    rmse: float
    n_quotes: int
    converged: bool
    n_starts: int


_DEFAULT_BOUNDS = (
    (0.01, 10.0),  # kappa
    (1e-4, 1.0),  # theta
    (1e-4, 3.0),  # sigma_v
    (-0.99, 0.99),  # rho
    (1e-4, 1.0),  # v0
)

_DEFAULT_STARTS = (
    (1.0, 0.04, 0.3, -0.5, 0.04),
    (2.0, 0.06, 0.6, -0.7, 0.06),
    (0.5, 0.03, 0.2, 0.0, 0.03),
    (3.0, 0.08, 1.0, -0.3, 0.08),
)

_PARAMETER_NAMES = ("kappa", "theta", "sigma_v", "rho", "v0")


def _price_rmse(
    parameters,
    spot: float,
    quotes,
    risk_free_rate: float,
    dividend_yield: float,
    upper_limit: float,
    grid_points: int,
) -> float:
    strikes = np.array([quote.strike for quote in quotes], dtype=float)
    tenors = np.array(
        [quote.time_to_expiry for quote in quotes],
        dtype=float,
    )
    option_types = [quote.option_type for quote in quotes]
    markets = np.array([quote.market_price for quote in quotes], dtype=float)
    kappa, theta, sigma_v, rho, v0 = parameters
    with np.errstate(all="ignore"):
        prices = heston_prices_semi_analytic(
            spot=spot,
            strikes=strikes,
            time_to_expiries=tenors,
            option_types=option_types,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            v0=v0,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            upper_limit=upper_limit,
            grid_points=grid_points,
        )
    if not np.all(np.isfinite(prices)):
        return 1e12
    return float(np.sqrt(np.mean((prices - markets) ** 2)))


def calibrate_heston(
    spot: float,
    quotes,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
    bounds=None,
    initial_guesses=None,
    max_iterations: int = 200,
    upper_limit: float = 40.0,
    grid_points: int = 4_000,
) -> HestonCalibrationResult:
    """Fit (kappa, theta, sigma_v, rho, v0) to European option quotes."""
    quotes = list(quotes)
    if len(quotes) < 5:
        raise ValueError("At least 5 quotes are required for calibration.")
    bounds = bounds if bounds is not None else _DEFAULT_BOUNDS
    initial_guesses = (
        initial_guesses
        if initial_guesses is not None
        else _DEFAULT_STARTS
    )

    def objective(parameters):
        return _price_rmse(
            parameters,
            spot,
            quotes,
            risk_free_rate,
            dividend_yield,
            upper_limit,
            grid_points,
        )

    best_parameters = None
    best_rmse = float("inf")
    any_converged = False
    for start in initial_guesses:
        result = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": max_iterations, "ftol": 1e-10},
        )
        if result.success:
            any_converged = True
        rmse = float(result.fun)
        if rmse < best_rmse:
            best_rmse = rmse
            best_parameters = result.x

    if best_parameters is None:
        best_parameters = np.asarray(initial_guesses[0], dtype=float)
    parameters = {
        name: float(value)
        for name, value in zip(_PARAMETER_NAMES, best_parameters)
    }
    return HestonCalibrationResult(
        parameters=parameters,
        rmse=best_rmse,
        n_quotes=len(quotes),
        converged=any_converged,
        n_starts=len(initial_guesses),
    )
