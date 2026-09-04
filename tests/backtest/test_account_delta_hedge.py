import pandas as pd

from src.backtest.account_delta_hedge import (
    AccountHedgeConfig,
    run_account_delta_hedge,
    run_rolling_account_straddle_backtest,
    run_account_straddle_backtest,
    save_account_hedge_result,
)
from src.reporting.plots import (
    plot_cost_breakdown,
    plot_equity_and_drawdown,
)


def _flat_prices(periods: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=periods, freq="B")
    return pd.DataFrame({"Close": [100.0] * periods}, index=dates)


def test_flat_market_account_hedge_reconciles():
    data = _flat_prices()
    result = run_account_delta_hedge(
        price_data=data,
        entry_date=data.index[0],
        config=AccountHedgeConfig(
            expiry_days=10,
            volatility=0.25,
        ),
    )
    assert result["reconciliation_passed"] is True
    assert abs(result["reconciliation_difference"]) < 1e-6
    assert result["total_pnl"] < 0.0  # premium + costs with no market move
    assert not result["snapshots"].empty
    assert not result["fills"].empty
    assert "margin_used" in result["snapshots"].columns
    assert (result["snapshots"]["margin_used"] >= 0).all()


def test_save_account_hedge_result_writes_files(tmp_path):
    data = _flat_prices()
    result = run_account_delta_hedge(
        price_data=data,
        entry_date=data.index[0],
        config=AccountHedgeConfig(expiry_days=10),
    )
    output = save_account_hedge_result(result, tmp_path)
    assert (output / "account_snapshots.csv").exists()
    assert (output / "fills.csv").exists()
    assert (output / "summary.json").exists()


def test_flat_market_account_straddle_reconciles():
    data = _flat_prices()
    result = run_account_straddle_backtest(
        price_data=data,
        entry_date=data.index[0],
        config=AccountHedgeConfig(
            expiry_days=10,
            volatility=0.25,
        ),
    )
    assert result["reconciliation_passed"] is True
    assert abs(result["reconciliation_difference"]) < 1e-6
    assert result["total_pnl"] < 0.0
    assert not result["snapshots"].empty


def _random_walk(periods: int = 700) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(5)
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.004, periods))
    return pd.DataFrame({"Close": prices}, index=dates)


def test_rolling_account_straddle_all_trades_reconcile():
    data = _random_walk()
    result = run_rolling_account_straddle_backtest(
        price_data=data,
        config=AccountHedgeConfig(
            expiry_days=10,
            volatility=0.20,
        ),
        entry_spacing=20,
        minimum_history=252,
    )
    assert result["summary"]["number_of_trades"] >= 2
    assert result["summary"]["reconciliation_failures"] == 0
    assert result["summary"]["max_abs_reconciliation_difference"] < 1e-6
    assert not result["equity_curve"].empty


def test_reporting_plots_save_files(tmp_path):
    data = _random_walk(periods=120)
    result = run_account_straddle_backtest(
        price_data=data,
        entry_date=data.index[0],
        config=AccountHedgeConfig(expiry_days=5),
    )
    equity_path = plot_equity_and_drawdown(
        result["snapshots"],
        tmp_path / "equity.png",
    )
    cost_path = plot_cost_breakdown(
        result["fills"],
        tmp_path / "costs.png",
    )
    assert equity_path.exists()
    assert cost_path.exists()
