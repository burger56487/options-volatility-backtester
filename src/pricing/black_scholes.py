from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from typing import Literal

from scipy.stats import norm


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BlackScholesResult:
    """Black-Scholes option value and first-/second-order Greeks."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _validate_inputs(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
    option_type: OptionType,
) -> None:
    """Validate Black-Scholes model inputs."""
    for name, value in {
        "spot": spot,
        "strike": strike,
        "time_to_expiry": time_to_expiry,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "dividend_yield": dividend_yield,
    }.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if spot <= 0:
        raise ValueError("spot must be positive.")

    if strike <= 0:
        raise ValueError("strike must be positive.")

    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive.")

    if volatility <= 0:
        raise ValueError("volatility must be positive.")

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")

def _d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float,
) -> tuple[float, float]:
    """Calculate d1 and d2 under Black-Scholes-Merton."""
    sqrt_time = sqrt(time_to_expiry)

    d1 = (
        log(spot / strike)
        + (
            risk_free_rate
            - dividend_yield
            + 0.5 * volatility**2
        )
        * time_to_expiry
    ) / (volatility * sqrt_time)

    d2 = d1 - volatility * sqrt_time

    return d1, d2


def price_and_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> BlackScholesResult:
    """
    Price a European option under Black-Scholes-Merton.

    Parameters:
        spot: Current underlying price.
        strike: Option strike price.
        time_to_expiry: Time to expiry in years.
        risk_free_rate: Continuously compounded annual risk-free rate.
        volatility: Annualised implied volatility as a decimal.
        option_type: Either "call" or "put".
        dividend_yield: Continuously compounded annual dividend yield.

    Returns:
        BlackScholesResult containing option price and Greeks.

    Notes:
        Vega is quoted for a 1.00 change in volatility.
        Theta is quoted per year.
        Rho is quoted for a 1.00 change in interest rate.
    """
    _validate_inputs(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        option_type=option_type,
    )

    d1, d2 = _d1_d2(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    discount_dividend = exp(-dividend_yield * time_to_expiry)
    discount_rate = exp(-risk_free_rate * time_to_expiry)
    sqrt_time = sqrt(time_to_expiry)

    gamma = (
        discount_dividend
        * norm.pdf(d1)
        / (spot * volatility * sqrt_time)
    )

    vega = (
        spot
        * discount_dividend
        * norm.pdf(d1)
        * sqrt_time
    )

    if option_type == "call":
        price = (
            spot * discount_dividend * norm.cdf(d1)
            - strike * discount_rate * norm.cdf(d2)
        )

        delta = discount_dividend * norm.cdf(d1)

        theta = (
            -spot
            * discount_dividend
            * norm.pdf(d1)
            * volatility
            / (2.0 * sqrt_time)
            - risk_free_rate
            * strike
            * discount_rate
            * norm.cdf(d2)
            + dividend_yield
            * spot
            * discount_dividend
            * norm.cdf(d1)
        )

        rho = (
            strike
            * time_to_expiry
            * discount_rate
            * norm.cdf(d2)
        )

    else:
        price = (
            strike * discount_rate * norm.cdf(-d2)
            - spot * discount_dividend * norm.cdf(-d1)
        )

        delta = -discount_dividend * norm.cdf(-d1)

        theta = (
            -spot
            * discount_dividend
            * norm.pdf(d1)
            * volatility
            / (2.0 * sqrt_time)
            + risk_free_rate
            * strike
            * discount_rate
            * norm.cdf(-d2)
            - dividend_yield
            * spot
            * discount_dividend
            * norm.cdf(-d1)
        )

        rho = (
            -strike
            * time_to_expiry
            * discount_rate
            * norm.cdf(-d2)
        )

    return BlackScholesResult(
        price=float(price),
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )


def option_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Return only the Black-Scholes-Merton European option price."""
    return price_and_greeks(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type=option_type,
        dividend_yield=dividend_yield,
    ).price
