"""Core performance ratios on daily returns."""

from __future__ import annotations

import math


def annualised_sharpe(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    excess = clean - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(
    returns,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    excess = clean - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return 0.0
    return float(
        excess.mean() / downside.std(ddof=1)
        * math.sqrt(periods_per_year)
    )


def calmar_ratio(
    annual_return: float,
    max_dd: float,
) -> float:
    if max_dd >= 0:
        return 0.0
    return float(annual_return / abs(max_dd))
