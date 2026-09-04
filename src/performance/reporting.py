"""Assemble a performance report from daily equity."""

from __future__ import annotations

import pandas as pd

from src.performance.annualisation import annualised_return
from src.performance.bootstrap import sharpe_ci
from src.performance.drawdown import (
    max_drawdown,
    underwater_duration_days,
)
from src.performance.ratios import (
    annualised_sharpe,
    calmar_ratio,
    sortino_ratio,
)
from src.performance.returns import daily_returns
from src.performance.tail_risk import historical_var_cvar


def compute_performance_report(
    equity: pd.Series,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.0,
    evaluation_mode: str = "in_sample",
) -> dict:
    returns = daily_returns(equity)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    annual_return = annualised_return(
        total_return,
        len(equity),
        periods_per_year,
    )
    max_dd = max_drawdown(equity)
    report = {
        "evaluation_mode": evaluation_mode,
        "observations": int(len(returns)),
        "total_return": total_return,
        "annualized_return": annual_return,
        "annualized_sharpe": annualised_sharpe(
            returns,
            risk_free_rate,
            periods_per_year,
        ),
        "sortino": sortino_ratio(
            returns,
            risk_free_rate,
            periods_per_year,
        ),
        "max_drawdown": max_dd,
        "calmar": calmar_ratio(annual_return, max_dd),
        "underwater_duration_days": underwater_duration_days(
            equity
        ),
        "tail_risk": historical_var_cvar(returns),
        "sharpe_ci": sharpe_ci(
            returns,
            periods_per_year=periods_per_year,
        ),
    }
    return report
