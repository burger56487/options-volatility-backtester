"""Explicit information-timeline semantics for daily-frequency backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    OBSERVATION = "observation"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    VALUATION = "valuation"


@dataclass(frozen=True)
class DecisionTimeline:
    observation_end: datetime
    signal_time: datetime
    order_time: datetime
    fill_time: datetime
    valuation_time: datetime

    def validate(self) -> None:
        if not (
            self.observation_end
            < self.signal_time
            <= self.order_time
            <= self.fill_time
            <= self.valuation_time
        ):
            raise ValueError(
                "Invalid timeline: must satisfy observation_end < "
                "signal_time <= order_time <= fill_time <= "
                "valuation_time."
            )
