"""Unified pricing result with explicit Greek conventions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GreeksConvention:
    theta_unit: str = "per_year"
    vega_unit: str = "per_1.0_volatility"
    dividend_model: str = "continuous_yield"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DEFAULT_CONVENTION = GreeksConvention()


@dataclass(frozen=True)
class PricingResult:
    price: float
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    theta: float | None = None
    rho: float | None = None
    method: str = "unknown"
    convention: GreeksConvention = DEFAULT_CONVENTION

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["convention"] = self.convention.to_dict()
        return output
