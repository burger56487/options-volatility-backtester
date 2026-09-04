import pandas as pd

from src.backtest.account_delta_hedge import (
    AccountHedgeConfig,
    run_account_delta_hedge,
    run_account_straddle_backtest,
    save_account_hedge_result,
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
