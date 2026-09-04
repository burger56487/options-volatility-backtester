"""Commission models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommissionSchedule:
    per_share: float = 0.005
    per_contract: float = 0.65
    minimum: float = 1.0


def commission_for(
    quantity: float,
    multiplier: float,
    schedule: CommissionSchedule = CommissionSchedule(),
) -> float:
    """Commission depends on share/contract count, not notional."""
    units = quantity * multiplier
    if multiplier > 1.0:  # option-style per-contract pricing
        amount = abs(quantity) * schedule.per_contract
    else:
        amount = abs(units) * schedule.per_share
    return max(amount, schedule.minimum)
