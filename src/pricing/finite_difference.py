"""Finite-difference PDE solvers for European options in log-space.

Implements the theta scheme: theta=0 explicit, theta=1 fully implicit,
theta=0.5 Crank-Nicolson. A stability check guards the explicit scheme.
"""

from __future__ import annotations

import math

from src.domain.enums import OptionType
from src.numerics.tridiagonal import solve_tridiagonal

from .requests import PricingRequest
from .results import PricingResult


def _transform(request: PricingRequest) -> tuple[float, float, float, float]:
    return (
        request.risk_free_rate - request.dividend_yield
        - 0.5 * request.volatility**2,
        request.volatility**2 / 2.0,
        request.risk_free_rate,
        request.dividend_yield,
    )


def finite_difference_price(
    request: PricingRequest,
    theta: float = 0.5,
    x_steps: int = 400,
    time_steps: int = 200,
    spot_multiplier: float = 6.0,
) -> PricingResult:
    """Price a European call/put on an S = exp(x) grid."""
    if request.volatility is None:
        raise ValueError("Finite differences require a volatility.")
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in [0, 1].")

    mu, half_var, rate, dividend = _transform(request)
    sigma = request.volatility
    t = request.time_to_expiry
    log_s0 = math.log(request.spot)
    log_min = log_s0 - spot_multiplier * sigma
    log_max = log_s0 + spot_multiplier * sigma
    h = (log_max - log_min) / x_steps
    dt = t / time_steps

    if theta == 0.0:
        # Explicit scheme stability: dt <= h^2 / (sigma^2 + |mu| h)
        limit = h * h / max(sigma * sigma, 1e-12)
        if dt > 0.95 * limit:
            raise ValueError(
                "Explicit scheme violates the stability condition: "
                f"dt={dt:.2e} > {0.95 * limit:.2e}"
            )

    log_x = [log_min + i * h for i in range(x_steps + 1)]
    spots = [math.exp(value) for value in log_x]

    def payoff(spot: float) -> float:
        if request.option_type == OptionType.CALL:
            return max(spot - request.strike, 0.0)
        return max(request.strike - spot, 0.0)

    values = [payoff(spot) for spot in spots]
    alpha = half_var / (h * h)
    beta = mu / (2.0 * h)

    def boundary(spot: float, tau: float) -> float:
        discount = math.exp(-rate * tau)
        forward = spot * math.exp(-dividend * tau)
        if request.option_type == OptionType.CALL:
            return max(forward - request.strike * discount, 0.0)
        return max(request.strike * discount - forward, 0.0)

    for step in range(time_steps):
        tau = (step + 1) * dt
        a = dt * (alpha - beta)
        b = dt * (alpha + beta)

        if theta == 0.0:
            new_values = [0.0] * (x_steps + 1)
            for i in range(1, x_steps):
                new_values[i] = (
                    a * values[i - 1]
                    + (1.0 - (2.0 * alpha + rate) * dt) * values[i]
                    + b * values[i + 1]
                )
            new_values[0] = boundary(spots[0], tau)
            new_values[x_steps] = boundary(spots[x_steps], tau)
            values = new_values
            continue

        # theta-scheme implicit update: A v^{n+1} = B v^n + boundary terms.
        diag_a = 1.0 + theta * dt * (2.0 * alpha + rate)
        upper_a = [-theta * b] * x_steps
        lower_a = [-theta * a] * x_steps

        diag_b = 1.0 - (1.0 - theta) * dt * (2.0 * alpha + rate)
        upper_b = [(1.0 - theta) * a] * x_steps
        lower_b = [(1.0 - theta) * b] * x_steps

        interior = x_steps - 1
        lower = [lower_a[i] for i in range(interior - 1)]
        diag = [diag_a] * interior
        upper = [upper_a[i] for i in range(interior - 1)]
        rhs = [0.0] * interior
        for i in range(interior):
            rhs[i] = (
                lower_b[i] * values[i]
                + diag_b * values[i + 1]
                + upper_b[i] * values[i + 2]
            )
        rhs[0] += theta * a * boundary(spots[0], tau)
        rhs[interior - 1] += theta * b * boundary(
            spots[x_steps], tau
        )
        rhs[0] += (1.0 - theta) * a * values[0]
        rhs[interior - 1] += (
            (1.0 - theta) * b * values[x_steps]
        )

        solved = solve_tridiagonal(lower, diag, upper, rhs)
        new_values = [0.0] * (x_steps + 1)
        new_values[1:x_steps] = solved
        new_values[0] = boundary(spots[0], tau)
        new_values[x_steps] = boundary(spots[x_steps], tau)
        values = new_values

    index = int(round((log_s0 - log_min) / h))
    index = max(1, min(x_steps - 1, index))
    price = values[index]
    return PricingResult(
        price=price,
        method=(
            "explicit_fd"
            if theta == 0.0
            else "implicit_fd"
            if theta == 1.0
            else "crank_nicolson_fd"
        ),
    )
