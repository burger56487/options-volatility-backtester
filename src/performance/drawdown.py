"""Drawdown and underwater-period metrics."""

from __future__ import annotations

import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_peak = equity.cummax()
    return equity / running_peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def underwater_duration_days(equity: pd.Series) -> dict[str, float]:
    """Mean / median / max consecutive calendar days underwater."""
    below_zero = drawdown_series(equity) < 0
    if not below_zero.any():
        return {"mean": 0.0, "median": 0.0, "max": 0.0}
    runs = []
    current = 0
    previous_date = None
    for index, value in below_zero.items():
        if value:
            current += (
                (index - previous_date).days
                if previous_date is not None
                else 1
            )
        else:
            if current > 0:
                runs.append(current)
            current = 0
        previous_date = index
    if current > 0:
        runs.append(current)
    return {
        "mean": float(sum(runs) / len(runs)),
        "median": float(sorted(runs)[len(runs) // 2]),
        "max": float(max(runs)),
    }
