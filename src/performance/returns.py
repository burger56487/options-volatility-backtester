"""Daily return series construction."""

from __future__ import annotations

import pandas as pd


def daily_returns(
    equity: pd.Series,
    equity_column: str = "equity",
) -> pd.Series:
    """Percent-change returns with zero return on non-trading first day."""
    frame = equity.copy()
    if isinstance(frame, pd.DataFrame):
        frame = frame[equity_column]
    returns = frame.pct_change().fillna(0.0)
    returns.name = "daily_return"
    return returns


def excess_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    aligned = pd.concat(
        [strategy_returns, benchmark_returns],
        axis=1,
        join="inner",
    )
    return (aligned.iloc[:, 0] - aligned.iloc[:, 1]).dropna()
