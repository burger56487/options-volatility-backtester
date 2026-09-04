"""Executed fills.

Convention: ``price`` is the actual execution price. Cash flow deducts only
the commission; spread/slippage/market-impact are attribution fields and are
NOT deducted again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .identifiers import InstrumentId
from .orders import Side


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    run_id: str
    timestamp: datetime
    instrument_id: InstrumentId
    side: Side
    quantity: float
    price: float
    multiplier: float
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    market_impact_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Fill quantity must be positive.")
        if self.price < 0:
            raise ValueError("Fill price cannot be negative.")
        if self.multiplier <= 0:
            raise ValueError("Multiplier must be positive.")

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.side == Side.BUY else -self.quantity

    @property
    def gross_notional(self) -> float:
        return self.quantity * self.price * self.multiplier

    @property
    def explicit_cost(self) -> float:
        return (
            self.commission
            + self.spread_cost
            + self.slippage_cost
            + self.market_impact_cost
        )

    @property
    def cash_flow(self) -> float:
        trading_cash_flow = (
            -self.gross_notional
            if self.side == Side.BUY
            else self.gross_notional
        )
        return trading_cash_flow - self.commission
