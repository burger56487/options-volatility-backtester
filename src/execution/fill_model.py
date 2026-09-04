"""Execution-price model under the actual-price convention.

The execution price already embeds half-spread, slippage and market impact.
Attribution fields (spread/slippage/impact) are recorded separately and are
NOT deducted again from cash; only the commission is charged on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.portfolio.fills import Fill
from src.portfolio.identifiers import InstrumentId
from src.portfolio.orders import Order, Side
from .market_snapshot import MarketSnapshot


@dataclass(frozen=True)
class ExecutionPricing:
    price: float
    half_spread: float
    slippage_per_share: float
    impact_per_share: float


@dataclass(frozen=True)
class ExecutionParameters:
    slippage_bps: float = 1.0
    impact_coefficient: float = 0.0
    impact_exponent: float = 0.5
    max_quote_age_seconds: float = 60.0
    allow_partial_fill: bool = True


def execution_price_for_side(
    side: Side,
    snapshot: MarketSnapshot,
    slippage_bps: float,
    impact_coefficient: float,
    quantity: float,
    impact_exponent: float = 0.5,
) -> ExecutionPricing:
    """Actual execution price = mid adjusted by spread, slippage, impact."""
    direction = 1.0 if side == Side.BUY else -1.0
    half_spread = 0.5 * snapshot.spread
    slippage_per_share = snapshot.mid * slippage_bps / 10_000.0
    impact_per_share = (
        impact_coefficient
        * (abs(quantity) ** impact_exponent)
        if impact_coefficient > 0
        else 0.0
    )
    price = (
        snapshot.mid
        + direction * (half_spread + slippage_per_share + impact_per_share)
    )
    return ExecutionPricing(
        price=max(price, 0.0),
        half_spread=half_spread,
        slippage_per_share=slippage_per_share,
        impact_per_share=impact_per_share,
    )


def make_fill(
    order: Order,
    run_id: str,
    fill_id: str,
    timestamp: datetime,
    quantity: float,
    multiplier: float,
    snapshot: MarketSnapshot,
    parameters: ExecutionParameters,
    commission: float,
) -> Fill:
    pricing = execution_price_for_side(
        side=order.side,
        snapshot=snapshot,
        slippage_bps=parameters.slippage_bps,
        impact_coefficient=parameters.impact_coefficient,
        quantity=quantity,
        impact_exponent=parameters.impact_exponent,
    )
    units = quantity * multiplier
    return Fill(
        fill_id=fill_id,
        order_id=order.order_id,
        run_id=run_id,
        timestamp=timestamp,
        instrument_id=order.instrument_id,
        side=order.side,
        quantity=quantity,
        price=pricing.price,
        multiplier=multiplier,
        commission=commission,
        spread_cost=pricing.half_spread * units,
        slippage_cost=pricing.slippage_per_share * units,
        market_impact_cost=pricing.impact_per_share * units,
    )
