"""Lightweight event-driven market-making simulator and metrics."""

from __future__ import annotations

import numpy as np

from .policies import fixed_quote_offsets, inventory_skew_offsets


def fill_probability(quote_offset: float, spread_scale: float, kappa: float = 2.0) -> float:
    """Probability a counterparty order hits our quote at this offset."""
    distance = abs(quote_offset) / max(spread_scale, 1e-9)
    return float(np.exp(-kappa * distance))


def simulate_quotes(
    mid_path,
    half_spread: float,
    inventory: float = 0.0,
    skew_strength: float = 0.0,
    seed: int | None = None,
):
    """Simulate one path of quote fills and mark-to-market PnL."""
    mid = np.asarray(mid_path, dtype=float)
    rng = np.random.default_rng(seed)
    cash = 0.0
    fills = []
    for i in range(len(mid) - 1):
        bid_offset, ask_offset = (
            fixed_quote_offsets(half_spread)
            if skew_strength == 0.0
            else inventory_skew_offsets(
                half_spread,
                inventory,
                max_inventory=1.0,
                skew_strength=skew_strength,
            )
        )
        # Market orders arrive one per step on a random side.
        side = "buy" if rng.random() < 0.5 else "sell"
        offset = ask_offset if side == "buy" else bid_offset
        if rng.random() < fill_probability(offset, half_spread):
            price = mid[i] + offset
            if side == "buy":  # they buy from us -> we short
                cash += price
                inventory -= 1.0
            else:
                cash -= price
                inventory += 1.0
            fills.append((i, side, price))
    terminal_pnl = cash + inventory * mid[-1]
    return {
        "terminal_pnl": float(terminal_pnl),
        "inventory": float(inventory),
        "fill_count": len(fills),
        "fills": fills,
    }


def market_making_metrics(results: dict) -> dict:
    """Extract headline metrics from a simulator result."""
    return {
        "terminal_pnl": results["terminal_pnl"],
        "final_inventory": results["inventory"],
        "fill_count": results["fill_count"],
    }
