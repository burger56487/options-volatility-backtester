"""Pricing engine registry with dispatch."""

from __future__ import annotations

from src.pricing.black_scholes import price_and_greeks

from .binomial import crr_price
from .finite_difference import finite_difference_price
from .monte_carlo import monte_carlo_price
from .requests import PricingRequest
from .results import PricingResult


def _black_scholes(request: PricingRequest) -> PricingResult:
    if request.volatility is None:
        raise ValueError("Black-Scholes requires a volatility.")
    result = price_and_greeks(
        spot=request.spot,
        strike=request.strike,
        time_to_expiry=request.time_to_expiry,
        risk_free_rate=request.risk_free_rate,
        volatility=request.volatility,
        option_type=request.option_type.value,
        dividend_yield=request.dividend_yield,
    )
    return PricingResult(
        price=result.price,
        delta=result.delta,
        gamma=result.gamma,
        vega=result.vega,
        theta=result.theta,
        rho=result.rho,
        method="black_scholes",
    )


def _crank_nicolson(request: PricingRequest) -> PricingResult:
    return finite_difference_price(request, theta=0.5)


def _monte_carlo(request: PricingRequest) -> PricingResult:
    mc = monte_carlo_price(request)
    return PricingResult(price=mc["price"], method="monte_carlo")


_ENGINES = {
    "black_scholes": _black_scholes,
    "crr": crr_price,
    "crank_nicolson": _crank_nicolson,
    "monte_carlo": _monte_carlo,
}


def price(request: PricingRequest, method: str) -> PricingResult:
    if method not in _ENGINES:
        raise KeyError(f"Unknown pricing method: {method}")
    return _ENGINES[method](request)


def available_methods() -> list[str]:
    return sorted(_ENGINES)
