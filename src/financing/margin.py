"""Research-grade simplified margin estimates (not exchange rules)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarginEstimate:
    initial_margin: float
    maintenance_margin: float
    methodology: str


def estimate_long_option_margin(
    premium_market_value: float,
) -> MarginEstimate:
    amount = max(0.0, premium_market_value)
    return MarginEstimate(
        initial_margin=amount,
        maintenance_margin=amount,
        methodology="long_option_premium_paid",
    )


def estimate_short_option_margin(
    option_market_value: float,
    underlying_market_value: float,
    out_of_the_money_amount: float,
    base_rate: float = 0.20,
    minimum_rate: float = 0.10,
) -> MarginEstimate:
    premium = abs(option_market_value)
    underlying = abs(underlying_market_value)
    initial = premium + max(
        base_rate * underlying - max(out_of_the_money_amount, 0.0),
        minimum_rate * underlying,
    )
    return MarginEstimate(
        initial_margin=initial,
        maintenance_margin=0.80 * initial,
        methodology="simplified_research_margin_model",
    )
