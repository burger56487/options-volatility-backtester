"""Shared quote-spread rules used by cleaning, validation and grading.

The wide-spread test is deliberately tick-aware: a one-tick wide quote on a
penny-priced option (e.g. 0.01 / 0.02) has a 100% relative spread yet is
normal market microstructure, so the rule requires the absolute spread to
exceed one tick as well as the relative threshold.
"""

from __future__ import annotations

import pandas as pd


DEFAULT_MAX_RELATIVE_SPREAD = 0.5
DEFAULT_MINIMUM_ABSOLUTE_SPREAD = 0.01


def wide_spread_mask(
    spread,
    mid,
    *,
    max_relative_spread: float = DEFAULT_MAX_RELATIVE_SPREAD,
    minimum_absolute_spread: float = DEFAULT_MINIMUM_ABSOLUTE_SPREAD,
) -> pd.Series:
    """Return a boolean mask for quotes whose spread is abnormally wide."""
    if max_relative_spread <= 0:
        raise ValueError("max_relative_spread must be positive.")
    if minimum_absolute_spread < 0:
        raise ValueError("minimum_absolute_spread must be non-negative.")
    spread = pd.Series(spread, dtype=float)
    mid = pd.Series(mid, dtype=float)
    return (
        (mid > 0)
        & (spread > minimum_absolute_spread)
        & (spread / mid > max_relative_spread)
    )
