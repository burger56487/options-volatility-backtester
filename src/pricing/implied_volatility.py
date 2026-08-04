from __future__ import annotations

from math import exp
from typing import Literal

from src.pricing.black_scholes import (
    OptionType,
    price_and_greeks,
)


IVMethod = Literal["bisection", "newton"]


def _validate_market_price(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: OptionType,
) -> None:
    """Check whether an option price lies inside no-arbitrage bounds."""
    if market_price < 0:
        raise ValueError("market_price must be non-negative.")

    discounted_spot = spot * exp(
        -dividend_yield * time_to_expiry
    )
    discounted_strike = strike * exp(
        -risk_free_rate * time_to_expiry
    )

    if option_type == "call":
        lower_bound = max(
            0.0,
            discounted_spot - discounted_strike,
        )
        upper_bound = discounted_spot
    else:
        lower_bound = max(
            0.0,
            discounted_strike - discounted_spot,
        )
        upper_bound = discounted_strike

    tolerance = 1e-10

    if market_price < lower_bound - tolerance:
        raise ValueError(
            "market_price is below the no-arbitrage lower bound."
        )

    if market_price > upper_bound + tolerance:
        raise ValueError(
            "market_price is above the no-arbitrage upper bound."
        )


def implied_volatility_bisection(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    lower_volatility: float = 1e-6,
    upper_volatility: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> float:
    """
    Recover Black-Scholes implied volatility using bisection.

    Volatility is returned as an annualised decimal, for example
    0.20 represents 20% annual volatility.
    """
    _validate_market_price(
        market_price=market_price,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
    )

    if lower_volatility <= 0:
        raise ValueError("lower_volatility must be positive.")

    if upper_volatility <= lower_volatility:
        raise ValueError(
            "upper_volatility must exceed lower_volatility."
        )

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    lower_price = price_and_greeks(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=lower_volatility,
        option_type=option_type,
        dividend_yield=dividend_yield,
    ).price

    upper_price = price_and_greeks(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=upper_volatility,
        option_type=option_type,
        dividend_yield=dividend_yield,
    ).price

    if market_price < lower_price - tolerance:
        raise ValueError(
            "market_price is below the price implied by lower_volatility."
        )

    if market_price > upper_price + tolerance:
        raise ValueError(
            "market_price exceeds the price implied by upper_volatility."
        )

    low = lower_volatility
    high = upper_volatility

    for _ in range(max_iterations):
        midpoint = (low + high) / 2.0

        model_price = price_and_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=midpoint,
            option_type=option_type,
            dividend_yield=dividend_yield,
        ).price

        pricing_error = model_price - market_price

        if abs(pricing_error) < tolerance:
            return midpoint

        if pricing_error < 0:
            low = midpoint
        else:
            high = midpoint

    raise RuntimeError(
        "Bisection did not converge within max_iterations."
    )


def implied_volatility_newton(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    initial_volatility: float = 0.20,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    minimum_vega: float = 1e-10,
) -> float:
    """
    Recover Black-Scholes implied volatility using Newton-Raphson.

    The routine falls back to bisection if the Newton update becomes
    numerically unstable or leaves the valid volatility range.
    """
    _validate_market_price(
        market_price=market_price,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
    )

    if initial_volatility <= 0:
        raise ValueError("initial_volatility must be positive.")

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    if minimum_vega <= 0:
        raise ValueError("minimum_vega must be positive.")

    volatility = initial_volatility

    for _ in range(max_iterations):
        result = price_and_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )

        pricing_error = result.price - market_price

        if abs(pricing_error) < tolerance:
            return volatility

        if abs(result.vega) < minimum_vega:
            break

        next_volatility = (
            volatility - pricing_error / result.vega
        )

        if next_volatility <= 0 or next_volatility > 5.0:
            break

        volatility = next_volatility

    return implied_volatility_bisection(
        market_price=market_price,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        option_type=option_type,
        dividend_yield=dividend_yield,
        tolerance=tolerance,
    )


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    method: IVMethod = "newton",
) -> float:
    """
    Recover Black-Scholes implied volatility.

    Supported methods are:
    - ``newton``: Newton-Raphson with bisection fallback;
    - ``bisection``: robust bisection solver.
    """
    if method == "newton":
        return implied_volatility_newton(
            market_price=market_price,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )

    if method == "bisection":
        return implied_volatility_bisection(
            market_price=market_price,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )

    raise ValueError(
        "method must be either 'newton' or 'bisection'."
    )
