from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Literal

import numpy as np
import pandas as pd

from src.pricing.black_scholes import (
    OptionType,
    price_and_greeks,
)


OptionSide = Literal["call", "put"]


@dataclass(frozen=True)
class VolatilitySurfaceParameters:
    """
    Parameters defining a transparent synthetic implied-volatility surface.

    Parameters are expressed in decimal volatility units. For example, 0.03
    represents a three-percentage-point volatility adjustment.
    """

    variance_risk_premium: float = 0.02
    term_structure_slope: float = 0.015
    smile_curvature: float = 0.12
    put_skew: float = -0.08
    minimum_volatility: float = 0.05
    option_spread_bps: float = 0.015
    minimum_spread: float = 0.01


def blended_realised_volatility(
    realized_vol_20d: float,
    realized_vol_60d: float,
    realized_vol_252d: float,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """
    Combine short-, medium-, and long-horizon realised-volatility estimates.

    Default weights emphasise the 20-day measure while reducing sensitivity to
    short-term volatility spikes through longer-horizon components.
    """
    if len(weights) != 3:
        raise ValueError("weights must contain exactly three values.")

    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative.")

    if not np.isclose(sum(weights), 1.0):
        raise ValueError("weights must sum to 1.0.")

    volatilities = [
        realized_vol_20d,
        realized_vol_60d,
        realized_vol_252d,
    ]

    if any(
        not np.isfinite(volatility) or volatility <= 0
        for volatility in volatilities
    ):
        raise ValueError(
            "realised volatilities must be finite and positive."
        )

    return float(
        sum(
            weight * volatility
            for weight, volatility in zip(
                weights,
                volatilities,
                strict=True,
            )
        )
    )


def synthetic_implied_volatility(
    spot: float,
    strike: float,
    time_to_expiry: float,
    base_volatility: float,
    option_type: OptionSide,
    parameters: VolatilitySurfaceParameters = (
        VolatilitySurfaceParameters()
    ),
) -> float:
    """
    Generate synthetic implied volatility using a stylised volatility surface.

    The surface includes:
    - a variance risk premium;
    - upward term structure with maturity;
    - smile curvature for deep ITM/OTM strikes;
    - negative skew that raises lower-strike put volatility.
    """
    if spot <= 0:
        raise ValueError("spot must be positive.")

    if strike <= 0:
        raise ValueError("strike must be positive.")

    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive.")

    if base_volatility <= 0:
        raise ValueError("base_volatility must be positive.")

    if option_type not in {"call", "put"}:
        raise ValueError(
            "option_type must be either 'call' or 'put'."
        )

    if parameters.minimum_volatility <= 0:
        raise ValueError(
            "minimum_volatility must be positive."
        )

    log_moneyness = log(strike / spot)

    term_adjustment = (
        parameters.term_structure_slope
        * np.sqrt(time_to_expiry)
    )

    smile_adjustment = (
        parameters.smile_curvature
        * log_moneyness**2
    )

    skew_adjustment = (
        parameters.put_skew * log_moneyness
    )

    implied_volatility = (
        base_volatility
        + parameters.variance_risk_premium
        + term_adjustment
        + smile_adjustment
        + skew_adjustment
    )

    return float(
        max(
            implied_volatility,
            parameters.minimum_volatility,
        )
    )


def create_synthetic_option_chain(
    valuation_date: pd.Timestamp,
    spot: float,
    base_volatility: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    days_to_expiry: tuple[int, ...] = (30, 60, 90, 180),
    strike_multipliers: tuple[float, ...] = (
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
    ),
    parameters: VolatilitySurfaceParameters = (
        VolatilitySurfaceParameters()
    ),
) -> pd.DataFrame:
    """
    Create a synthetic European option chain with bid/ask prices and Greeks.

    Each requested maturity and strike is generated for both call and put
    options. Theoretical mid prices come from Black-Scholes-Merton pricing;
    bid/ask quotes are created using a transparent percentage spread rule.
    """
    if pd.isna(valuation_date):
        raise ValueError("valuation_date must not be missing.")

    if spot <= 0:
        raise ValueError("spot must be positive.")

    if base_volatility <= 0:
        raise ValueError("base_volatility must be positive.")

    if not days_to_expiry:
        raise ValueError("days_to_expiry must not be empty.")

    if not strike_multipliers:
        raise ValueError("strike_multipliers must not be empty.")

    if any(days <= 0 for days in days_to_expiry):
        raise ValueError(
            "days_to_expiry values must be positive."
        )

    if any(multiplier <= 0 for multiplier in strike_multipliers):
        raise ValueError(
            "strike_multipliers must be positive."
        )

    if parameters.option_spread_bps < 0:
        raise ValueError(
            "option_spread_bps must be non-negative."
        )

    records: list[dict[str, float | int | str | pd.Timestamp]] = []

    for days in sorted(days_to_expiry):
        time_to_expiry = days / 365.0
        expiry_date = valuation_date + pd.Timedelta(days=days)

        for multiplier in sorted(strike_multipliers):
            strike = round(spot * multiplier, 2)

            for option_type in ("call", "put"):
                implied_volatility = synthetic_implied_volatility(
                    spot=spot,
                    strike=strike,
                    time_to_expiry=time_to_expiry,
                    base_volatility=base_volatility,
                    option_type=option_type,
                    parameters=parameters,
                )

                pricing_result = price_and_greeks(
                    spot=spot,
                    strike=strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=implied_volatility,
                    option_type=option_type,
                    dividend_yield=dividend_yield,
                )

                mid_price = pricing_result.price

                half_spread = max(
                    mid_price * parameters.option_spread_bps / 2.0,
                    parameters.minimum_spread / 2.0,
                )

                bid_price = max(mid_price - half_spread, 0.0)
                ask_price = mid_price + half_spread

                records.append(
                    {
                        "valuation_date": valuation_date,
                        "expiry_date": expiry_date,
                        "days_to_expiry": days,
                        "time_to_expiry": time_to_expiry,
                        "option_type": option_type,
                        "spot": spot,
                        "strike": strike,
                        "moneyness": strike / spot,
                        "log_moneyness": log(strike / spot),
                        "implied_volatility": implied_volatility,
                        "bid": bid_price,
                        "ask": ask_price,
                        "mid": mid_price,
                        "spread": ask_price - bid_price,
                        "delta": pricing_result.delta,
                        "gamma": pricing_result.gamma,
                        "vega": pricing_result.vega,
                        "theta": pricing_result.theta,
                        "rho": pricing_result.rho,
                    }
                )

    chain = pd.DataFrame(records)

    return chain.sort_values(
        [
            "days_to_expiry",
            "strike",
            "option_type",
        ]
    ).reset_index(drop=True)


def select_atm_straddle(
    chain: pd.DataFrame,
    days_to_expiry: int,
) -> pd.DataFrame:
    """
    Select the nearest-to-ATM call and put for a given maturity.

    Returns exactly two rows when a valid call-put pair exists.
    """
    required_columns = {
        "days_to_expiry",
        "strike",
        "spot",
        "option_type",
    }

    missing_columns = required_columns - set(chain.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    eligible = chain.loc[
        chain["days_to_expiry"] == days_to_expiry
    ].copy()

    if eligible.empty:
        raise ValueError(
            f"No options found for {days_to_expiry} days to expiry."
        )

    eligible["atm_distance"] = (
        eligible["strike"] - eligible["spot"]
    ).abs()

    atm_strike = eligible.loc[
        eligible["atm_distance"].idxmin(),
        "strike",
    ]

    straddle = eligible.loc[
        eligible["strike"] == atm_strike
    ].copy()

    option_types = set(straddle["option_type"])

    if option_types != {"call", "put"}:
        raise ValueError(
            "ATM selection requires both a call and a put."
        )

    return straddle.sort_values(
        "option_type"
    ).reset_index(drop=True)
