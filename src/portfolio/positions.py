"""Position accounting with weighted average cost and realised PnL."""

from __future__ import annotations

from dataclasses import dataclass

from .fills import Fill
from .identifiers import InstrumentId


@dataclass
class Position:
    instrument_id: InstrumentId
    quantity: float = 0.0
    average_cost: float = 0.0
    multiplier: float = 1.0
    realised_pnl: float = 0.0

    def apply_fill(self, fill: Fill) -> None:
        if fill.instrument_id != self.instrument_id:
            raise ValueError("Fill instrument does not match position.")

        old_quantity = self.quantity
        trade_quantity = fill.signed_quantity
        new_quantity = old_quantity + trade_quantity
        same_direction = (
            old_quantity == 0 or old_quantity * trade_quantity > 0
        )

        if same_direction:
            old_notional = abs(old_quantity) * self.average_cost
            trade_notional = abs(trade_quantity) * fill.price
            total_quantity = abs(old_quantity) + abs(trade_quantity)
            self.average_cost = (
                (old_notional + trade_notional) / total_quantity
                if total_quantity > 0
                else 0.0
            )
        else:
            closing_quantity = min(
                abs(old_quantity),
                abs(trade_quantity),
            )
            direction = 1.0 if old_quantity > 0 else -1.0
            self.realised_pnl += (
                direction
                * closing_quantity
                * (fill.price - self.average_cost)
                * self.multiplier
            )
            if abs(trade_quantity) > abs(old_quantity):
                self.average_cost = fill.price
            elif new_quantity == 0:
                self.average_cost = 0.0

        self.quantity = new_quantity
        self.multiplier = fill.multiplier

    def market_value(self, market_price: float) -> float:
        return self.quantity * market_price * self.multiplier

    def unrealised_pnl(self, market_price: float) -> float:
        return (
            self.quantity
            * (market_price - self.average_cost)
            * self.multiplier
        )
