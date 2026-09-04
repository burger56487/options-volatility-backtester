"""Paper PnL evaluation for strategy definitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from src.domain.enums import OptionType
from src.pricing.black_scholes import price_and_greeks
from src.strategy_lib.contracts import StrategyDefinition


def value_strategy(
    definition: StrategyDefinition,
    spot: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
    days_held: int = 0,
) -> float:
    """Market value of a strategy's legs after holding for ``days_held``."""
    multiplier = 100.0
    total = 0.0
    for leg in definition.legs:
        strike = spot * (1.0 + leg.strike_offset_pct)
        remaining_days = max(leg.days_to_expiry - days_held, 0)
        result = price_and_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=max(remaining_days / 365.0, 1e-8),
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=leg.option_type.value,
            dividend_yield=dividend_yield,
        )
        total += leg.signed_quantity * result.price * multiplier
    return float(total)


def paper_strategy_pnl(
    definition: StrategyDefinition,
    entry_spot: float,
    exit_spot: float,
    days_held: int,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """Entry-to-exit PnL at a flat implied volatility (research illustration)."""
    entry_value = value_strategy(
        definition,
        entry_spot,
        risk_free_rate,
        volatility,
        dividend_yield,
        days_held=0,
    )
    exit_value = value_strategy(
        definition,
        exit_spot,
        risk_free_rate,
        volatility,
        dividend_yield,
        days_held=days_held,
    )
    return exit_value - entry_value
