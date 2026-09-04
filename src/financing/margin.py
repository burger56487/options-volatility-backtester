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


def estimate_account_margin(
    account,
    market_prices: dict[str, float],
    underlying_spots: dict[str, float] | None = None,
) -> dict:
    """Estimate total margin used by an account under the research model.

    Long options use paid premium; short options use the simplified short
    margin (OTM relief assumed zero without more data); short stock uses its
    market value. Not exchange margin rules.
    """
    from src.portfolio.identifiers import InstrumentType

    underlying_spots = underlying_spots or {}
    initial_total = 0.0
    maintenance_total = 0.0
    details = []
    for key, position in account.positions.items():
        instrument = position.instrument_id
        market_value = position.market_value(market_prices[key])
        if instrument.instrument_type == InstrumentType.OPTION:
            if position.quantity >= 0:
                estimate = estimate_long_option_margin(market_value)
            else:
                spot = underlying_spots.get(
                    instrument.symbol,
                    abs(market_value) / max(abs(position.quantity), 1e-12)
                    / position.multiplier,
                )
                estimate = estimate_short_option_margin(
                    option_market_value=market_value,
                    underlying_market_value=spot,
                    out_of_the_money_amount=0.0,
                )
        elif instrument.instrument_type == InstrumentType.STOCK:
            amount = max(0.0, abs(market_value))
            estimate = MarginEstimate(
                initial_margin=amount,
                maintenance_margin=amount,
                methodology="short_stock_market_value",
            )
        else:
            continue
        initial_total += estimate.initial_margin
        maintenance_total += estimate.maintenance_margin
        details.append(
            {
                "instrument": key,
                "initial_margin": estimate.initial_margin,
                "maintenance_margin": estimate.maintenance_margin,
                "methodology": estimate.methodology,
            }
        )
    return {
        "initial_margin_total": initial_total,
        "maintenance_margin_total": maintenance_total,
        "details": details,
    }
