"""Static no-arbitrage screens: butterfly and calendar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pricing.black_scholes import option_price


def check_butterfly_calls(
    quotes: pd.DataFrame,
    risk_free_rate: float,
    dividend_yield: float = 0.012,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Return butterfly violations for call mid prices at one expiry."""
    rows = []
    for expiry, group in quotes.groupby("expiry"):
        calls = group[group["option_type"] == "call"].sort_values(
            "strike"
        )
        for i in range(1, len(calls) - 1):
            up, mid, down = (
                calls.iloc[i + 1],
                calls.iloc[i],
                calls.iloc[i - 1],
            )
            butterfly = 0.5 * (
                float(up["mid"]) + float(down["mid"])
            ) - float(mid["mid"])
            if butterfly < -tolerance:
                rows.append(
                    {
                        "expiry": expiry,
                        "strike": mid["strike"],
                        "butterfly_value": float(butterfly),
                        "violation": True,
                    }
                )
    return pd.DataFrame(rows)


def check_calendar_screen(
    surface_points: pd.DataFrame,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Flag pairs where total variance falls as expiry lengthens."""
    rows = []
    expiries = sorted(surface_points["expiry"].unique())
    for i in range(len(expiries) - 1):
        t1_frame = surface_points[
            surface_points["expiry"] == expiries[i]
        ]
        t2_frame = surface_points[
            surface_points["expiry"] == expiries[i + 1]
        ]
        common = pd.merge(
            t1_frame,
            t2_frame,
            on="log_moneyness",
            suffixes=("_1", "_2"),
        )
        for _, row in common.iterrows():
            if row["total_variance_2"] < row["total_variance_1"] - tolerance:
                rows.append(
                    {
                        "expiry_1": expiries[i],
                        "expiry_2": expiries[i + 1],
                        "log_moneyness": row["log_moneyness"],
                        "total_variance_1": row["total_variance_1"],
                        "total_variance_2": row["total_variance_2"],
                        "violation": True,
                    }
                )
    return pd.DataFrame(rows)
