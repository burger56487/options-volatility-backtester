"""Execution quality metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.portfolio.fills import Fill


@dataclass(frozen=True)
class ExecutionMetrics:
    fill_rate: float
    total_commission: float
    total_spread_cost: float
    total_slippage_cost: float
    total_impact_cost: float

    def to_dict(self) -> dict:
        return asdict(self)


def execution_metrics(
    requested_quantity: float,
    fills: list[Fill],
) -> ExecutionMetrics:
    filled_quantity = sum(fill.quantity for fill in fills)
    return ExecutionMetrics(
        fill_rate=(
            filled_quantity / requested_quantity
            if requested_quantity > 0
            else 0.0
        ),
        total_commission=sum(fill.commission for fill in fills),
        total_spread_cost=sum(fill.spread_cost for fill in fills),
        total_slippage_cost=sum(fill.slippage_cost for fill in fills),
        total_impact_cost=sum(
            fill.market_impact_cost for fill in fills
        ),
    )


def fills_to_dataframe(fills: list[Fill]) -> pd.DataFrame:
    rows = []
    for fill in fills:
        rows.append(
            {
                "fill_id": fill.fill_id,
                "order_id": fill.order_id,
                "timestamp": fill.timestamp.isoformat(),
                "instrument": fill.instrument_id.key(),
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.price,
                "commission": fill.commission,
                "spread_cost": fill.spread_cost,
                "slippage_cost": fill.slippage_cost,
                "impact_cost": fill.market_impact_cost,
                "cash_flow": fill.cash_flow,
            }
        )
    return pd.DataFrame(rows)
