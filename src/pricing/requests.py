"""Unified pricing request."""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.enums import ExerciseStyle, OptionType


@dataclass(frozen=True)
class PricingRequest:
    spot: float
    strike: float
    time_to_expiry: float
    risk_free_rate: float
    option_type: OptionType
    dividend_yield: float = 0.0
    volatility: float | None = None
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    steps: int = 500

    def __post_init__(self) -> None:
        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("Spot and strike must be positive.")
        if self.time_to_expiry < 0:
            raise ValueError("Time to expiry cannot be negative.")
        if self.volatility is not None and self.volatility < 0:
            raise ValueError("Volatility cannot be negative.")
