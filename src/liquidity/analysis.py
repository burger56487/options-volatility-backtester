"""Lightweight chain-liquidity state assessment.

This is a deliberately simple stage-9 placeholder used by the end-to-end
pipeline: it grades the snapshot by median relative spread, the share of
quotes with meaningful open interest/volume and the quality-good share.
It will be superseded by the full liquidity step when that spec lands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiquidityState:
    overall_state: str
    median_rel_spread: float
    pct_reliable: float
    pct_good: float
    n_quotes: int


def assess_liquidity_state(df: pd.DataFrame) -> LiquidityState:
    """Return an overall liquidity state for a full quote snapshot."""
    work = df.copy()
    if "mid" not in work.columns and {"bid", "ask"}.issubset(
        work.columns
    ):
        work["mid"] = 0.5 * (work["bid"] + work["ask"])
    if "rel_spread" not in work.columns and {"bid", "ask"}.issubset(
        work.columns
    ):
        work["rel_spread"] = (
            (work["ask"] - work["bid"]) / work["mid"]
        )

    n = int(len(work))
    if n == 0:
        return LiquidityState(
            overall_state="no_data",
            median_rel_spread=float("nan"),
            pct_reliable=0.0,
            pct_good=0.0,
            n_quotes=0,
        )

    median_rel_spread = float(
        pd.to_numeric(work.get("rel_spread"), errors="coerce").median()
    )
    if np.isnan(median_rel_spread):
        median_rel_spread = float("nan")

    reliable = 0
    if {"volume", "open_interest"}.issubset(work.columns):
        volume = pd.to_numeric(work["volume"], errors="coerce").fillna(0)
        oi = pd.to_numeric(
            work["open_interest"],
            errors="coerce",
        ).fillna(0)
        reliable = int(((volume > 0) | (oi > 0)).sum())
    pct_reliable = 100.0 * reliable / n if n else 0.0

    pct_good = 100.0
    if "quality" in work.columns:
        pct_good = 100.0 * float(
            (work["quality"] == "good").mean()
        )

    if np.isfinite(median_rel_spread):
        if median_rel_spread <= 0.05 and pct_reliable >= 50:
            state = "good"
        elif median_rel_spread <= 0.20:
            state = "moderate"
        else:
            state = "poor"
    else:
        state = "unknown"

    return LiquidityState(
        overall_state=state,
        median_rel_spread=median_rel_spread,
        pct_reliable=pct_reliable,
        pct_good=pct_good,
        n_quotes=n,
    )
