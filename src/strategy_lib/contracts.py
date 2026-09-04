"""Strategy lifecycle contracts: definitions, inputs, outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.domain.enums import OptionType
from src.pricing.requests import PricingRequest


@dataclass(frozen=True)
class LegSpec:
    """One option leg of a strategy."""

    option_type: OptionType
    strike_offset_pct: float  # e.g. 0.0 ATM, -0.05, +0.05
    days_to_expiry: int
    signed_quantity: int = 1  # +1 long, -1 short


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    legs: list[LegSpec]
    delta_neutral: bool = False


@dataclass(frozen=True)
class LegQuote:
    instrument_key: str
    price: float
    multiplier: float
    delta: float  # per contract
    gamma: float  # per contract
    vega: float  # per contract
    theta: float  # per contract
    rho: float  # per contract


@dataclass(frozen=True)
class StrategyInput:
    spot: float
    valuation_date: date
    quotes: dict[str, LegQuote] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyOutput:
    strategy: str
    premium: float
    gross_notional: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "premium": self.premium,
            "gross_notional": self.gross_notional,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


def leg_key(
    underlying: str,
    expiry: date,
    strike: float,
    option_type: OptionType,
) -> str:
    return (
        f"{underlying}:{expiry.isoformat()}:{strike:.2f}:"
        f"{option_type.value}"
    )


def pricing_request_for_leg(
    spot: float,
    strike: float,
    expiry: date,
    valuation_date: date,
    risk_free_rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    volatility: float | None = None,
) -> PricingRequest:
    return PricingRequest(
        spot=spot,
        strike=strike,
        time_to_expiry=max(
            (expiry - valuation_date).days / 365.0,
            1e-8,
        ),
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
        option_type=option_type,
    )
