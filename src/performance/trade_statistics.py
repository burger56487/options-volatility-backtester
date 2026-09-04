"""Trade-level statistics and portfolio turnover."""

from __future__ import annotations

import pandas as pd


def trade_statistics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trade_count": 0,
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "average_pnl": float("nan"),
        }
    pnl = trades["final_pnl"]
    wins = pnl[pnl > 0]
    losses = -pnl[pnl < 0]
    return {
        "trade_count": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": (
            float(wins.sum() / losses.sum())
            if losses.sum() > 0
            else float("inf")
        ),
        "average_pnl": float(pnl.mean()),
    }


def portfolio_turnover(
    total_traded_notional: float,
    average_equity: float,
) -> float:
    if average_equity <= 0:
        return float("nan")
    return float(total_traded_notional / average_equity)
