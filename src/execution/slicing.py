"""Order slicing under participation-rate constraints."""

from __future__ import annotations


def slice_order(
    quantity: float,
    volume: float,
    max_participation_rate: float,
    slice_cap: float | None = None,
) -> list[float]:
    """Split an order into executable slices respecting participation."""
    if quantity <= 0:
        return []
    if volume <= 0 or not 0 < max_participation_rate <= 1:
        raise ValueError(
            "volume must be positive and participation in (0, 1]."
        )
    max_slice = volume * max_participation_rate
    if slice_cap is not None:
        max_slice = min(max_slice, slice_cap)
    if max_slice <= 0:
        return []
    slices = []
    remaining = quantity
    while remaining > 0:
        take = min(max_slice, remaining)
        slices.append(take)
        remaining -= take
    return slices


def opportunity_cost_estimate(
    unfilled_quantity: float,
    mid_price: float,
    slippage_bps: float,
) -> float:
    """Estimated cost of leaving an order unfilled at the current mid."""
    return abs(unfilled_quantity) * mid_price * slippage_bps / 10_000.0
