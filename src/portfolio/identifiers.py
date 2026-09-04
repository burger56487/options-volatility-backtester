"""Unified instrument identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from src.domain.enums import ExerciseStyle, OptionType


class InstrumentType(str, Enum):
    STOCK = "stock"
    OPTION = "option"
    CASH = "cash"


@dataclass(frozen=True)
class InstrumentId:
    instrument_type: InstrumentType
    symbol: str
    expiry: date | None = None
    strike: float | None = None
    option_type: OptionType | None = None

    def key(self) -> str:
        if self.instrument_type == InstrumentType.STOCK:
            return f"STOCK:{self.symbol}"
        if self.instrument_type == InstrumentType.OPTION:
            if (
                self.expiry is None
                or self.strike is None
                or self.option_type is None
            ):
                raise ValueError(
                    "Option identifier missing expiry, strike or type."
                )
            return (
                f"OPTION:{self.symbol}:{self.expiry.isoformat()}:"
                f"{self.strike:.6f}:{self.option_type.value}"
            )
        return f"{self.instrument_type.value}:{self.symbol}"
