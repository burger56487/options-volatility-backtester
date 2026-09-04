import numpy as np
import pandas as pd

from src.risk.limits import RiskLimits, limit_usage_report
from src.risk.measure import (
    filtered_historical_var,
    monte_carlo_var,
)
from src.risk.scenarios import preset_scenario_shocks, stress_loss


def _pnl():
    rng = np.random.default_rng(9)
    return pd.Series(rng.normal(0, 100, 300))


def test_filtered_historical_var_runs():
    result = filtered_historical_var(_pnl())
    assert result.insufficient_sample is False
    assert result.var > 0


def test_monte_carlo_var_uses_simulator():
    def simulator(n, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 100, n)

    result = monte_carlo_var(simulator, n_scenarios=5_000, seed=1)
    assert result.insufficient_sample is False
    assert 100 < result.var < 500


def test_surface_stress_scenario_and_usage_report():
    exposures = {"equity": 100.0, "volatility": 200.0, "skew": 300.0}
    loss = stress_loss(
        exposures,
        preset_scenario_shocks("vol_surface_stress"),
    )
    assert loss["total"] < 0
    limits = RiskLimits(
        max_gross_exposure=1000.0,
        max_leverage=2.0,
        max_abs_delta=100.0,
        max_abs_gamma=50.0,
        max_abs_vega=500.0,
        max_daily_loss=1000.0,
        max_drawdown=0.2,
        min_cash_buffer=100.0,
    )
    report = limit_usage_report(
        current={
            "gross_exposure": 800.0,
            "leverage": 1.0,
            "delta": 50.0,
            "gamma": 10.0,
            "vega": 300.0,
            "cash": 500.0,
        },
        limits=limits,
    )
    assert any(row["limit"] == "max_gross_exposure" for row in report)
