"""Hedging policies and risk-neutralisation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class HedgeState:
    spot: float
    delta: float  # option portfolio delta in shares
    gamma: float  # per dollar-spot change in delta
    vega: float
    theta: float
    current_shares: float


class HedgingPolicy(Protocol):
    def target_shares(self, state: HedgeState) -> float:
        ...


class FixedDeltaHedge:
    """Rebalance fully to -delta at every decision point."""

    def target_shares(self, state: HedgeState) -> float:
        return -state.delta


class ThresholdDeltaHedge:
    """Rebalance only when |current + delta| exceeds the band."""

    def __init__(self, band_shares: float = 10.0) -> None:
        self.band_shares = band_shares

    def target_shares(self, state: HedgeState) -> float:
        exposure = state.current_shares + state.delta
        if abs(exposure) > self.band_shares:
            return -state.delta
        return state.current_shares


def cost_aware_band(
    gamma: float,
    spot: float,
    cost_per_share: float,
    volatility: float,
    dt: float,
) -> float:
    """Approximate exposure band balancing gamma benefit against cost.

    band ~ (3 * cost_per_share / (gamma * spot^2))^(1/3), a standard
    discrete-hedging heuristic; volatility/time enter only as diagnostics.
    """
    if gamma <= 0 or spot <= 0 or cost_per_share < 0:
        raise ValueError("gamma/spot must be positive and cost non-negative.")
    return float((3.0 * cost_per_share / (gamma * spot**2)) ** (1.0 / 3.0))


def neutralize_delta_gamma(
    delta_portfolio: float,
    gamma_portfolio: float,
    delta_secondary_option: float,
    gamma_secondary_option: float,
):
    """Solve for stock shares and secondary option quantity to flatten both."""
    denominator = delta_secondary_option * gamma_secondary_option
    if abs(denominator) < 1e-12:
        raise ValueError("Secondary option must carry both delta and gamma.")
    # Solve [1 delta_opt; 0 gamma_opt] [q_stock; q_opt] = -[delta; gamma]
    q_option = -gamma_portfolio / gamma_secondary_option
    q_stock = -(delta_portfolio + q_option * delta_secondary_option)
    return q_stock, q_option
