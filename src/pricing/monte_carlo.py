"""Monte Carlo pricing with variance-reduction diagnostics."""

from __future__ import annotations

import math

import numpy as np

from src.domain.enums import OptionType

from .requests import PricingRequest


def _payoff(terminal: np.ndarray, request: PricingRequest) -> np.ndarray:
    if request.option_type == OptionType.CALL:
        return np.maximum(terminal - request.strike, 0.0)
    return np.maximum(request.strike - terminal, 0.0)


def monte_carlo_price(
    request: PricingRequest,
    n_paths: int = 100_000,
    seed: int | None = None,
    antithetic: bool = True,
    control_variate: bool = True,
) -> dict:
    """Price a European option and report a variance-reduction ratio."""
    if request.volatility is None:
        raise ValueError("Monte Carlo requires a volatility.")
    if request.time_to_expiry <= 0:
        raise ValueError("Monte Carlo requires positive time to expiry.")

    discount = math.exp(-request.risk_free_rate * request.time_to_expiry)
    z = np.random.default_rng(seed).standard_normal(n_paths)
    if antithetic:
        z = np.concatenate([z, -z])

    terminal = request.spot * np.exp(
        (
            request.risk_free_rate
            - request.dividend_yield
            - 0.5 * request.volatility**2
        )
        * request.time_to_expiry
        + request.volatility
        * math.sqrt(request.time_to_expiry)
        * z
    )
    payoffs = discount * _payoff(terminal, request)

    if control_variate:
        expected_terminal = request.spot * math.exp(
            (
                request.risk_free_rate
                - request.dividend_yield
            )
            * request.time_to_expiry
        )
        raw = _payoff(terminal, request)
        var_terminal = float(np.var(terminal, ddof=1))
        beta = (
            float(np.cov(raw, terminal, ddof=1)[0, 1])
            / var_terminal
            if var_terminal > 0
            else 0.0
        )
        adjusted = payoffs - discount * beta * (
            terminal - expected_terminal
        )
        variance_ratio = float(
            np.var(payoffs, ddof=1) / np.var(adjusted, ddof=1)
        )
        payoffs = adjusted
    else:
        variance_ratio = 1.0

    price = float(np.mean(payoffs))
    se = float(np.std(payoffs, ddof=1) / math.sqrt(len(payoffs)))
    return {
        "price": price,
        "standard_error": se,
        "ci_low": price - 1.96 * se,
        "ci_high": price + 1.96 * se,
        "variance_reduction_ratio": variance_ratio,
        "n_paths": int(len(payoffs)),
    }
