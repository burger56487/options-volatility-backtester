"""Generic finite-difference Greeks with analytic consistency checks."""

from __future__ import annotations

from dataclasses import replace

from src.pricing.black_scholes import price_and_greeks

from .requests import PricingRequest


def analytic_result(request: PricingRequest):
    if request.volatility is None:
        raise ValueError("Volatility required.")
    return price_and_greeks(
        spot=request.spot,
        strike=request.strike,
        time_to_expiry=request.time_to_expiry,
        risk_free_rate=request.risk_free_rate,
        volatility=request.volatility,
        option_type=request.option_type.value,
        dividend_yield=request.dividend_yield,
    )


def finite_difference_greeks(
    request: PricingRequest,
    bump: float = 1e-4,
    time_bump: float = 1e-3,
) -> dict[str, float]:
    """Central-difference Greeks for European BSM pricing."""
    def price(r: PricingRequest) -> float:
        return analytic_result(r).price

    base = analytic_result(request)
    up_spot = price(replace(request, spot=request.spot + bump))
    down_spot = price(replace(request, spot=request.spot - bump))
    up_vol = price(
        replace(request, volatility=request.volatility + bump)
    )
    down_vol = price(
        replace(request, volatility=request.volatility - bump)
    )
    up_time = price(
        replace(
            request,
            time_to_expiry=request.time_to_expiry + time_bump,
        )
    )
    down_time = price(
        replace(
            request,
            time_to_expiry=request.time_to_expiry - time_bump,
        )
    )
    return {
        "delta": (up_spot - down_spot) / (2 * bump),
        "gamma": (up_spot - 2 * base.price + down_spot) / bump**2,
        "vega": (up_vol - down_vol) / (2 * bump),
        # theta = -dV/d(tau); the bump increases time to expiry.
        "theta": -(up_time - down_time) / (2 * time_bump),
        "rho": 0.0,
    }


def greeks_consistency_report(
    request: PricingRequest,
) -> dict[str, float]:
    """Compare analytic Greeks against finite-difference estimates."""
    analytic = analytic_result(request)
    numeric = finite_difference_greeks(request)
    return {
        "delta_diff": abs(analytic.delta - numeric["delta"]),
        "gamma_diff": abs(analytic.gamma - numeric["gamma"]),
        "vega_diff": abs(analytic.vega - numeric["vega"]),
        "theta_diff": abs(analytic.theta - numeric["theta"]),
    }
