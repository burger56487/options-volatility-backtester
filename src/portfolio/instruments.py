"""Contract definitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .identifiers import (
    ExerciseStyle,
    InstrumentId,
    InstrumentType,
    OptionType,
)


@dataclass(frozen=True)
class Stock:
    symbol: str
    currency: str = "USD"
    multiplier: float = 1.0

    @property
    def instrument_id(self) -> InstrumentId:
        return InstrumentId(
            instrument_type=InstrumentType.STOCK,
            symbol=self.symbol,
        )


@dataclass(frozen=True)
class OptionContract:
    underlying_symbol: str
    expiry: date
    strike: float
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    multiplier: int = 100
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError("Strike must be positive.")
        if self.multiplier <= 0:
            raise ValueError("Multiplier must be positive.")

    @property
    def instrument_id(self) -> InstrumentId:
        return InstrumentId(
            instrument_type=InstrumentType.OPTION,
            symbol=self.underlying_symbol,
            expiry=self.expiry,
            strike=self.strike,
            option_type=self.option_type,
        )
