"""Convergence studies with observed-order estimates."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .binomial import crr_price
from .requests import PricingRequest


def convergence_study(
    request: PricingRequest,
    reference_price: float,
    engine,
    step_name: str,
    steps: list[int],
) -> pd.DataFrame:
    """Return error and observed error ratio for increasing steps."""
    rows = []
    previous_error = None
    for step in steps:
        current = engine(request, step)
        error = abs(current - reference_price)
        row = {
            step_name: step,
            "price": current,
            "absolute_error": error,
        }
        if previous_error is not None and error > 0:
            row["observed_order"] = (
                previous_error / error
                if previous_error > 0
                else float("nan")
            )
        else:
            row["observed_order"] = float("nan")
        rows.append(row)
        previous_error = error
    return pd.DataFrame(rows)


def crr_with_steps(request: PricingRequest, steps: int) -> float:
    return crr_price(replace(request, steps=steps)).price
