"""Market snapshot with orderbook or mid-only modes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MarketSnapshot:
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    mid: float
    bid_size: float | None = None
    ask_size: float | None = None
    volume: float | None = None
    orderbook_mode: str = "mid_only"
    quote_age_seconds: float = 0.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def relative_spread(self) -> float:
        return self.spread / self.mid if self.mid > 0 else float("inf")

    def is_stale(self, max_quote_age_seconds: float) -> bool:
        return self.quote_age_seconds > max_quote_age_seconds
