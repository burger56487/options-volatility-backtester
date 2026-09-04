"""Heston stochastic volatility: characteristic function and MC pricing."""

from __future__ import annotations

import cmath
import math

import numpy as np

from src.domain.enums import OptionType


def heston_characteristic_function(
    u: complex,
    time_to_expiry: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
) -> complex:
    """Characteristic function E[exp(i*u*X_T)] with X_T = ln S_T/S_0.

    Uses the formulation with a = kappa*theta and the log-branch handled by
    cmath.log. Assumes zero risk-free and dividend inside the log process for
    the identity checks; calibration code should add the drift separately.
    """
    i = 1j
    tau = time_to_expiry
    d = cmath.sqrt(
        (rho * sigma_v * i * u - kappa) ** 2
        + sigma_v**2 * (i * u + u**2)
    )
    g = (kappa - rho * sigma_v * i * u - d) / (
        kappa - rho * sigma_v * i * u + d
    )
    exp_d = cmath.exp(-d * tau)
    denominator = 1.0 - g * exp_d
    c_term = (
        kappa
        * theta
        / sigma_v**2
        * (
            (kappa - rho * sigma_v * i * u - d) * tau
            - 2.0 * cmath.log(denominator)
        )
    )
    d_term = (
        (kappa - rho * sigma_v * i * u - d)
        / sigma_v**2
        * (1.0 - exp_d)
        / denominator
    )
    return cmath.exp(c_term + d_term * v0)


def _simulate_heston_paths(
    spot: float,
    time_to_expiry: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
    risk_free_rate: float,
    dividend_yield: float,
    n_paths: int,
    n_steps: int,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-truncation Euler simulation of S and v."""
    rng = np.random.default_rng(seed)
    dt = time_to_expiry / max(n_steps, 1)
    log_s = np.full(n_paths, math.log(spot))
    v = np.full(n_paths, max(v0, 0.0))
    drift = risk_free_rate - dividend_yield - 0.5 * v
    sqrt_dt = math.sqrt(dt)
    for _ in range(max(n_steps, 1)):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w_s = z1
        w_v = rho * z1 + math.sqrt(1.0 - rho**2) * z2
        v_pos = np.maximum(v, 0.0)
        v = (
            v_pos
            + kappa * (theta - v_pos) * dt
            + sigma_v * np.sqrt(v_pos) * sqrt_dt * w_v
        )
        drift = risk_free_rate - dividend_yield - 0.5 * np.maximum(
            v, 0.0
        )
        log_s = (
            log_s
            + drift * dt
            + np.sqrt(np.maximum(v, 0.0)) * sqrt_dt * w_s
        )
    return np.exp(log_s), np.maximum(v, 0.0)


def heston_mc_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    n_paths: int = 40_000,
    n_steps: int = 60,
    seed: int | None = None,
) -> dict:
    """Price a European option under Heston with full truncation."""
    terminal, _ = _simulate_heston_paths(
        spot=spot,
        time_to_expiry=time_to_expiry,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        v0=v0,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )
    discount = math.exp(-risk_free_rate * time_to_expiry)
    if option_type == OptionType.CALL:
        payoffs = np.maximum(terminal - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal, 0.0)
    prices = discount * payoffs
    price = float(np.mean(prices))
    se = float(np.std(prices, ddof=1) / math.sqrt(len(prices)))
    return {
        "price": price,
        "standard_error": se,
        "ci_low": price - 1.96 * se,
        "ci_high": price + 1.96 * se,
    }
