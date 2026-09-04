"""Capital-budget based position sizing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SizingMethod(str, Enum):
    FIXED_CONTRACTS = "fixed_contracts"
    FIXED_PREMIUM_BUDGET = "fixed_premium_budget"
    FIXED_VEGA_BUDGET = "fixed_vega_budget"
    FIXED_GAMMA_BUDGET = "fixed_gamma_budget"


@dataclass(frozen=True)
class PositionSizingRequest:
    method: SizingMethod
    equity: float
    premium_per_contract: float = 0.0
    premium_budget_fraction: float = 0.02
    fixed_contracts: int = 1
    vega_budget: float | None = None
    vega_per_contract: float | None = None
    gamma_budget: float | None = None
    gamma_per_contract: float | None = None
    multiplier: float = 100.0


def straddle_premium(
    call_premium_per_share: float,
    put_premium_per_share: float,
    multiplier: float = 100.0,
) -> float:
    return (call_premium_per_share + put_premium_per_share) * multiplier


def calculate_contract_quantity(
    request: PositionSizingRequest,
) -> int:
    if request.equity <= 0:
        return 0
    if request.method == SizingMethod.FIXED_CONTRACTS:
        return max(0, request.fixed_contracts)
    if request.method == SizingMethod.FIXED_PREMIUM_BUDGET:
        if request.premium_per_contract <= 0:
            raise ValueError("Premium per contract required.")
        return max(
            0,
            math.floor(
                request.equity
                * request.premium_budget_fraction
                / request.premium_per_contract
            ),
        )
    if request.method == SizingMethod.FIXED_VEGA_BUDGET:
        if (
            request.vega_budget is None
            or request.vega_per_contract is None
            or request.vega_per_contract == 0
        ):
            raise ValueError("Vega budget parameters required.")
        return max(
            0,
            math.floor(
                abs(request.vega_budget / request.vega_per_contract)
            ),
        )
    if request.method == SizingMethod.FIXED_GAMMA_BUDGET:
        if (
            request.gamma_budget is None
            or request.gamma_per_contract is None
            or request.gamma_per_contract == 0
        ):
            raise ValueError("Gamma budget parameters required.")
        return max(
            0,
            math.floor(
                abs(
                    request.gamma_budget
                    / request.gamma_per_contract
                )
            ),
        )
    raise ValueError(f"Unsupported sizing method: {request.method}")
