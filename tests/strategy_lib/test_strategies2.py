from datetime import date

import pandas as pd

from src.domain.enums import OptionType
from src.strategy_lib.exit import ExitPlan, evaluate_exit
from src.strategy_lib.signals import (
    realized_vol_forecast_from_past,
    vrp_signal,
)
from src.strategy_lib.strategies2 import (
    calendar_spread,
    gamma_scalping_pnl,
    neutral_delta_quantity,
    risk_reversal,
    run_strategy_survey,
)


def test_calendar_spread_delta_uses_same_side():
    output = run_strategy_survey(
        [calendar_spread()],
        spot=100.0,
        valuation_date=date(2026, 9, 4),
        risk_free_rate=0.04,
    )
    assert len(output) == 1


def test_risk_reversal_survey_and_neutral():
    frame = run_strategy_survey(
        [risk_reversal()],
        spot=100.0,
        valuation_date=date(2026, 9, 4),
        risk_free_rate=0.04,
    )
    row = frame.iloc[0]
    assert row["hedge_shares"] == -row["delta"]


def test_gamma_scalping_pnl_sign():
    # Realized below implied => negative gamma PnL for a long-gamma book.
    pnl = gamma_scalping_pnl(
        gamma=10.0,
        spot=100.0,
        realized_vol=0.15,
        implied_vol=0.25,
        dt=1 / 252,
    )
    assert pnl < 0


def test_exit_manager_triggers():
    plan = ExitPlan(max_days=30, target_pnl_pct=0.05, stop_loss_pct=0.02)
    assert evaluate_exit(plan, days_held=2, pnl_pct=0.06) == "target_pnl"
    assert evaluate_exit(plan, days_held=2, pnl_pct=-0.03) == "stop_loss"
    assert evaluate_exit(plan, days_held=31, pnl_pct=0.0) == "max_days"
    assert evaluate_exit(plan, days_held=1, pnl_pct=0.01) is None


def test_vrp_signal_never_uses_future_returns():
    returns = pd.Series([0.01] * 100, index=pd.bdate_range("2026-01-01", periods=100))
    forecast = realized_vol_forecast_from_past(returns, window=10)
    iv = pd.Series(0.2, index=forecast.index)
    signal = vrp_signal(iv, forecast, threshold=1.0)
    assert signal.dtype == bool
    # Insert a future shock after day 90; forecasts before 90 must be equal.
    shocked = returns.copy()
    shocked.iloc[95:] = shocked.iloc[95:] * 100
    forecast_shocked = realized_vol_forecast_from_past(shocked, window=10)
    assert forecast.iloc[:90].equals(forecast_shocked.iloc[:90])
