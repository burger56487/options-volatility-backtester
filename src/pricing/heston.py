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
    log_ratio = cmath.log(denominator / (1.0 - g))
    c_term = (
        kappa
        * theta
        / sigma_v**2
        * (
            (kappa - rho * sigma_v * i * u - d) * tau
            - 2.0 * log_ratio
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
    sqrt_dt = math.sqrt(dt)
    for _ in range(max(n_steps, 1)):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w_s = z1
        w_v = rho * z1 + math.sqrt(1.0 - rho**2) * z2
        v_pos = np.maximum(v, 0.0)
        # Full-truncation Euler: the log-price step uses the variance at the
        # start of the step, otherwise the correlated same-step shock biases
        # the drift by ~rho*sigma_v/2 per unit time.
        log_s = (
            log_s
            + (risk_free_rate - dividend_yield - 0.5 * v_pos) * dt
            + np.sqrt(v_pos) * sqrt_dt * w_s
        )
        v = (
            v_pos
            + kappa * (theta - v_pos) * dt
            + sigma_v * np.sqrt(v_pos) * sqrt_dt * w_v
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


def heston_price_semi_analytic(
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
    upper_limit: float = 60.0,
    grid_points: int = 20_000,
) -> float:
    """Price a European option under Heston via Fourier inversion.

    Uses the standard probability representation
    ``C = S e^{-qT} P1 - K e^{-rT} P2`` with ``Pj`` obtained from the
    characteristic function by Simpson integration on a dense u-grid. For a
    vanishing
    vol-of-vol (``sigma_v <= 1e-8``) the variance path is deterministic and
    the price falls back to the exact Gaussian limit, which avoids the
    cancellation errors of the closed-form CF in that regime.
    """
    prices = heston_prices_semi_analytic(
        spot=spot,
        strikes=np.array([strike], dtype=float),
        time_to_expiries=np.array([time_to_expiry], dtype=float),
        option_types=[option_type],
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
    return float(prices[0])


def heston_prices_semi_analytic(
    spot: float,
    strikes,
    time_to_expiries,
    option_types,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    upper_limit: float = 60.0,
    grid_points: int = 20_000,
) -> np.ndarray:
    """Vectorised European Heston prices on one shared Simpson grid."""
    from src.pricing.black_scholes import option_price

    strikes = np.asarray(strikes, dtype=float)
    time_to_expiries = np.asarray(time_to_expiries, dtype=float)
    if strikes.ndim != 1 or time_to_expiries.ndim != 1:
        raise ValueError("strikes and time_to_expiries must be 1-D arrays")
    n_quotes = len(strikes)
    if len(time_to_expiries) != n_quotes or len(option_types) != n_quotes:
        raise ValueError("quote arrays must have the same length")

    prices = np.empty(n_quotes, dtype=float)
    for index, (strike, tau, option_type) in enumerate(
        zip(strikes, time_to_expiries, option_types)
    ):
        if sigma_v <= 1e-8:
            decay = (
                (1.0 - math.exp(-kappa * tau)) / kappa
                if kappa > 1e-12
                else tau
            )
            integrated_variance = theta * tau + (v0 - theta) * decay
            total_variance = max(integrated_variance, 1e-16)
            prices[index] = option_price(
                spot=spot,
                strike=float(strike),
                time_to_expiry=float(tau),
                risk_free_rate=risk_free_rate,
                volatility=math.sqrt(total_variance / tau),
                option_type=option_type.value,
                dividend_yield=dividend_yield,
            )
            continue

        u = np.linspace(1e-8, upper_limit, grid_points + 1)
        k = math.log(float(strike) / spot)
        drift = (risk_free_rate - dividend_yield) * float(tau)
        base = _heston_cf_array(
            u,
            float(tau),
            kappa,
            theta,
            sigma_v,
            rho,
            v0,
        )
        f2 = np.exp(1j * u * drift) * base
        f2_neg_i = np.exp(drift) * _heston_cf_array(
            np.array([-1j]),
            float(tau),
            kappa,
            theta,
            sigma_v,
            rho,
            v0,
        )[0]
        u_minus = u - 1j
        base_minus = _heston_cf_array(
            u_minus,
            float(tau),
            kappa,
            theta,
            sigma_v,
            rho,
            v0,
        )
        f1 = np.exp(1j * u_minus * drift) * base_minus / f2_neg_i
        integrand2 = np.real(np.exp(-1j * u * k) * f2 / (1j * u))
        integrand1 = np.real(np.exp(-1j * u * k) * f1 / (1j * u))
        dx = u[1] - u[0]
        p2 = 0.5 + (1.0 / math.pi) * _simpson(integrand2, dx)
        p1 = 0.5 + (1.0 / math.pi) * _simpson(integrand1, dx)
        call = (
            spot * math.exp(-dividend_yield * float(tau)) * p1
            - float(strike)
            * math.exp(-risk_free_rate * float(tau))
            * p2
        )
        if option_type == OptionType.PUT:
            prices[index] = (
                call
                - spot * math.exp(-dividend_yield * float(tau))
                + float(strike)
                * math.exp(-risk_free_rate * float(tau))
            )
        else:
            prices[index] = call
    return prices


def _heston_cf_array(
    u,
    time_to_expiry: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
) -> np.ndarray:
    """Vectorised version of :func:`heston_characteristic_function`."""
    i = 1j
    tau = time_to_expiry
    d = np.sqrt(
        (rho * sigma_v * i * u - kappa) ** 2
        + sigma_v**2 * (i * u + u**2)
    )
    g = (kappa - rho * sigma_v * i * u - d) / (
        kappa - rho * sigma_v * i * u + d
    )
    exp_d = np.exp(-d * tau)
    denominator = 1.0 - g * exp_d
    log_ratio = np.log(denominator / (1.0 - g))
    c_term = (
        kappa
        * theta
        / sigma_v**2
        * (
            (kappa - rho * sigma_v * i * u - d) * tau
            - 2.0 * log_ratio
        )
    )
    d_term = (
        (kappa - rho * sigma_v * i * u - d)
        / sigma_v**2
        * (1.0 - exp_d)
        / denominator
    )
    return np.exp(c_term + d_term * v0)


def _simpson(y: np.ndarray, dx: float) -> float:
    """Composite Simpson's rule over a uniformly spaced grid."""
    n = len(y) - 1
    if n % 2 == 1:
        n -= 1
    total = y[0] + y[n]
    total += 4.0 * y[1:n:2].sum()
    total += 2.0 * y[2 : n - 1 : 2].sum()
    return float(total * dx / 3.0)
