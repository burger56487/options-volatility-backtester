"""CRR binomial tree for European and American options."""

from __future__ import annotations

import math

from src.domain.enums import ExerciseStyle, OptionType

from .requests import PricingRequest
from .results import PricingResult


def crr_price(request: PricingRequest) -> PricingResult:
    """Price with a CRR tree using continuous dividend yield."""
    if request.volatility is None:
        raise ValueError("Binomial pricing requires a volatility.")
    n = max(int(request.steps), 1)
    dt = request.time_to_expiry / n
    u = math.exp(request.volatility * math.sqrt(dt))
    d = 1.0 / u
    drift = math.exp(
        (request.risk_free_rate - request.dividend_yield) * dt
    )
    p = (drift - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError("Risk-neutral probability outside [0,1].")

    def payoff(spot: float) -> float:
        if request.option_type == OptionType.CALL:
            return max(spot - request.strike, 0.0)
        return max(request.strike - spot, 0.0)

    values = [
        payoff(request.spot * (u ** (n - i)) * (d ** i))
        for i in range(n + 1)
    ]
    discount = math.exp(-request.risk_free_rate * dt)
    american = request.exercise_style == ExerciseStyle.AMERICAN

    for step in range(n - 1, -1, -1):
        for i in range(step + 1):
            spot = request.spot * (u ** (step - i)) * (d ** i)
            value = discount * (
                p * values[i] + (1.0 - p) * values[i + 1]
            )
            if american:
                value = max(value, payoff(spot))
            values[i] = value
    return PricingResult(
        price=values[0],
        method=f"crr_binomial_n={n}",
    )
