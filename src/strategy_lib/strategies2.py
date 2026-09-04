"""Additional strategies and portfolio helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.domain.enums import OptionType

from .contracts import (
    LegSpec,
    StrategyDefinition,
)
from .strategies import price_strategy


def calendar_spread(
    short_days: int = 30,
    long_days: int = 60,
    strike_offset_pct: float = 0.0,
    option_type: OptionType = OptionType.CALL,
) -> StrategyDefinition:
    """Short near-dated leg, long far-dated leg, same strike."""
    return StrategyDefinition(
        name="calendar_spread",
        legs=[
            LegSpec(option_type, strike_offset_pct, short_days, -1),
            LegSpec(option_type, strike_offset_pct, long_days, +1),
        ],
    )


def risk_reversal(
    days_to_expiry: int = 30,
    put_offset: float = -0.05,
    call_offset: float = 0.05,
) -> StrategyDefinition:
    """Long OTM put, short OTM call (negative-skew carry view)."""
    return StrategyDefinition(
        name="risk_reversal",
        legs=[
            LegSpec(OptionType.PUT, put_offset, days_to_expiry, +1),
            LegSpec(OptionType.CALL, call_offset, days_to_expiry, -1),
        ],
    )


def neutral_delta_quantity(delta: float) -> float:
    """Shares of underlying needed to make the option portfolio delta-flat."""
    return float(-delta)


def gamma_scalping_pnl(
    gamma: float,
    spot: float,
    realized_vol: float,
    implied_vol: float,
    dt: float,
) -> float:
    """One-period gamma PnL proxy: 0.5*Gamma*S^2*(rv^2 - iv^2)*dt."""
    return 0.5 * gamma * spot**2 * (
        realized_vol**2 - implied_vol**2
    ) * dt


def run_strategy_survey(
    definitions: list[StrategyDefinition],
    spot: float,
    valuation_date: date,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    volatility: float = 0.25,
) -> pd.DataFrame:
    """Price several strategies and return one row per strategy."""
    rows = []
    for definition in definitions:
        output = price_strategy(
            definition,
            spot=spot,
            valuation_date=valuation_date,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
        )
        payload = output.to_dict()
        payload["hedge_shares"] = neutral_delta_quantity(output.delta)
        rows.append(payload)
    return pd.DataFrame(rows)
