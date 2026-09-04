"""Trade record with full timing-audit fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass
class TradeRecord:
    trade_id: str
    run_id: str
    symbol: str
    strategy_name: str

    observation_end: datetime
    signal_time: datetime
    order_time: datetime
    fill_time: datetime
    exit_time: datetime

    entry_spot: float
    exit_spot: float
    quantity: float
    entry_value: float
    exit_value: float
    transaction_cost: float
    net_pnl: float

    split_name: str
    evaluation_mode: str

    def validate_timeline(self) -> None:
        if not (
            self.observation_end
            < self.signal_time
            <= self.order_time
            <= self.fill_time
            < self.exit_time
        ):
            raise ValueError(
                f"Trade {self.trade_id} has an invalid timeline."
            )

    def to_dict(self) -> dict[str, Any]:
        self.validate_timeline()
        output = asdict(self)
        for field in [
            "observation_end",
            "signal_time",
            "order_time",
            "fill_time",
            "exit_time",
        ]:
            output[field] = output[field].isoformat()
        return output
