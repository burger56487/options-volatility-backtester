"""Unified exit manager."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExitPlan:
    max_days: int
    target_pnl_pct: float | None = None
    stop_loss_pct: float | None = None


def evaluate_exit(
    plan: ExitPlan,
    days_held: int,
    pnl_pct: float,
) -> str | None:
    """Return an exit reason if any condition triggers, else None."""
    if days_held >= plan.max_days:
        return "max_days"
    if plan.target_pnl_pct is not None and pnl_pct >= plan.target_pnl_pct:
        return "target_pnl"
    if plan.stop_loss_pct is not None and pnl_pct <= -abs(
        plan.stop_loss_pct
    ):
        return "stop_loss"
    return None
