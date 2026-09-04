"""Unified data structures for underlying bars and option quotes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class DataType(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class UnderlyingBar:
    trade_date: date
    symbol: str
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: float
    source: str
    data_type: DataType = DataType.REAL

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["trade_date"] = self.trade_date.isoformat()
        output["data_type"] = self.data_type.value
        return output


@dataclass(frozen=True)
class OptionQuote:
    timestamp: datetime
    underlying_symbol: str
    expiry: date
    strike: float
    option_type: OptionType
    bid: float
    ask: float
    spot: float
    risk_free_rate: float
    dividend_yield: float
    source: str
    data_type: DataType
    last: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    multiplier: int = 100
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def relative_spread(self) -> float:
        if self.mid <= 0:
            return float("inf")
        return self.spread / self.mid

    @property
    def maturity_days(self) -> int:
        return (self.expiry - self.timestamp.date()).days

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["timestamp"] = self.timestamp.isoformat()
        output["expiry"] = self.expiry.isoformat()
        output["option_type"] = self.option_type.value
        output["data_type"] = self.data_type.value
        output["exercise_style"] = self.exercise_style.value
        output["mid"] = self.mid
        output["spread"] = self.spread
        output["relative_spread"] = self.relative_spread
        output["maturity_days"] = self.maturity_days
        return output
