"""Quote policies: fixed, inventory-skewed, Greeks-aware, Avellaneda-Stoikov."""

from __future__ import annotations

import math


def fixed_quote_offsets(half_spread: float) -> tuple[float, float]:
    """Symmetrical quotes around the mid: (bid_offset, ask_offset)."""
    return -half_spread, half_spread


def inventory_skew_offsets(
    half_spread: float,
    inventory: float,
    max_inventory: float,
    skew_strength: float = 1.0,
) -> tuple[float, float]:
    """Shift both quotes down (long) or up (short) to work off inventory."""
    scale = max(abs(inventory) / max(max_inventory, 1.0), 0.0)
    # Long inventory: shift both quotes down to sell more and buy less.
    shift = -skew_strength * half_spread * scale * (
        1.0 if inventory > 0 else -1.0
    )
    return -half_spread + shift, half_spread + shift


def avellaneda_stoikov_offsets(
    inventory: float,
    risk_aversion: float,
    volatility: float,
    tau: float,
    order_intensity: float,
    mid: float = 0.0,
) -> tuple[float, float]:
    """A-S style reservation price and half-spread in price units."""
    gamma = risk_aversion
    variance_term = gamma * volatility**2 * tau
    reservation = mid - variance_term * inventory
    half_spread = variance_term + (2.0 / gamma) * math.log(
        1.0 + gamma / max(order_intensity, 1e-9)
    )
    return reservation - half_spread - mid, reservation + half_spread - mid
