"""Concrete multi-leg strategies and portfolio aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from src.domain.enums import OptionType
from src.pricing.black_scholes import price_and_greeks

from .contracts import (
    LegQuote,
    LegSpec,
    StrategyDefinition,
    StrategyInput,
    StrategyOutput,
)


def long_straddle(
    days_to_expiry: int = 30,
) -> StrategyDefinition:
    return StrategyDefinition(
        name="long_straddle",
        legs=[
            LegSpec(OptionType.CALL, 0.0, days_to_expiry, +1),
            LegSpec(OptionType.PUT, 0.0, days_to_expiry, +1),
        ],
    )


def long_strangle(
    days_to_expiry: int = 30,
    put_offset: float = -0.05,
    call_offset: float = 0.05,
) -> StrategyDefinition:
    return StrategyDefinition(
        name="long_strangle",
        legs=[
            LegSpec(OptionType.PUT, put_offset, days_to_expiry, +1),
            LegSpec(OptionType.CALL, call_offset, days_to_expiry, +1),
        ],
    )


def iron_butterfly(
    days_to_expiry: int = 30,
    wing_offset: float = 0.05,
) -> StrategyDefinition:
    return StrategyDefinition(
        name="iron_butterfly",
        legs=[
            LegSpec(OptionType.PUT, -wing_offset, days_to_expiry, +1),
            LegSpec(OptionType.PUT, 0.0, days_to_expiry, -1),
            LegSpec(OptionType.CALL, 0.0, days_to_expiry, -1),
            LegSpec(OptionType.CALL, wing_offset, days_to_expiry, +1),
        ],
    )


def price_strategy(
    definition: StrategyDefinition,
    spot: float,
    valuation_date: date,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    volatility: float = 0.25,
) -> StrategyOutput:
    """Price all legs with BSM and aggregate per-contract Greeks."""
    total = {
        "premium": 0.0,
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
        "rho": 0.0,
    }
    multiplier = 100.0
    for leg in definition.legs:
        strike = spot * (1.0 + leg.strike_offset_pct)
        expiry = valuation_date + timedelta(days=leg.days_to_expiry)
        result = price_and_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=max(leg.days_to_expiry / 365.0, 1e-8),
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=leg.option_type.value,
            dividend_yield=dividend_yield,
        )
        quantity = leg.signed_quantity
        total["premium"] += quantity * result.price * multiplier
        total["delta"] += quantity * result.delta * multiplier
        total["gamma"] += quantity * result.gamma * multiplier
        total["vega"] += quantity * result.vega * multiplier
        total["theta"] += quantity * result.theta * multiplier
        total["rho"] += quantity * result.rho * multiplier
    return StrategyOutput(
        strategy=definition.name,
        premium=float(total["premium"]),
        gross_notional=float(
            sum(
                abs(leg.signed_quantity)
                * multiplier
                * spot
                * (1.0 + abs(leg.strike_offset_pct))
                for leg in definition.legs
            )
        ),
        delta=float(total["delta"]),
        gamma=float(total["gamma"]),
        vega=float(total["vega"]),
        theta=float(total["theta"]),
        rho=float(total["rho"]),
    )
