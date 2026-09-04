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
    max_inventory: float | None = None,
    loss_limit: float | None = None,
    seed: int | None = None,
):
    """Simulate one path of quote fills and mark-to-market PnL."""
    mid = np.asarray(mid_path, dtype=float)
    rng = np.random.default_rng(seed)
    cash = 0.0
    fills = []
    anti_cheat_penalty = 0.0
    halted = False
    for i in range(len(mid) - 1):
        if loss_limit is not None and cash < -abs(loss_limit):
            halted = True
            break
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
        anti_cheat_penalty += (1.0 - fill_probability(offset, half_spread)) * (
            2.0 * half_spread
        )
        would_fill = rng.random() < fill_probability(offset, half_spread)
        if (
            would_fill
            and (
                max_inventory is None
                or abs(inventory + (1.0 if side == "sell" else -1.0))
                <= max_inventory
            )
        ):
            price = mid[i] + offset
            if side == "buy":  # they buy from us -> we short
                cash += price
                inventory -= 1.0
            else:
                cash -= price
                inventory += 1.0
            fills.append((i, side, price))
    if (
        max_inventory is not None
        and abs(inventory) > max_inventory
    ):
        cash += inventory * mid[-1]  # force liquidation at last mark
        inventory = 0.0
    terminal_pnl = cash + inventory * mid[-1]
    return {
        "terminal_pnl": float(terminal_pnl),
        "inventory": float(inventory),
        "fill_count": len(fills),
        "halted": bool(halted),
        "anti_cheat_penalty": float(anti_cheat_penalty),
        "fills": fills,
    }


def market_making_metrics(results: dict) -> dict:
    """Extract headline metrics from a simulator result."""
    return {
        "terminal_pnl": results["terminal_pnl"],
        "final_inventory": results["inventory"],
        "fill_count": results["fill_count"],
    }
