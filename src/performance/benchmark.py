"""Benchmark-relative statistics."""

from __future__ import annotations

import pandas as pd

from .ratios import annualised_sharpe
from .returns import excess_returns


def benchmark_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: float = 252.0,
) -> dict:
    excess = excess_returns(strategy_returns, benchmark_returns)
    aligned = pd.concat(
        [strategy_returns, benchmark_returns],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 5:
        return {"insufficient_sample": True}
    cov = aligned.cov().iloc[0, 1]
    var = aligned.iloc[:, 1].var()
    beta = float(cov / var) if var > 0 else float("nan")
    alpha_daily = float(
        aligned.iloc[:, 0].mean() - beta * aligned.iloc[:, 1].mean()
    )
    return {
        "insufficient_sample": False,
        "excess_sharpe": annualised_sharpe(
            excess,
            periods_per_year=periods_per_year,
        ),
        "beta": beta,
        "alpha_annualized": alpha_daily * periods_per_year,
        "n_observations": int(len(aligned)),
    }
