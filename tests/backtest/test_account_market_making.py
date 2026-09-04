"""Tests for the account-engine market-making integration."""

from __future__ import annotations

import pandas as pd

from src.backtest.account_market_making import (
    AccountMarketMakingConfig,
    default_market_making_limits,
    run_account_market_making,
    save_account_market_making_result,
)
from src.risk.limits import RiskLimits


def _flat_prices(periods: int = 45) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=periods, freq="B")
    return pd.DataFrame({"Close": [100.0] * periods}, index=dates)


def _gap_prices(
    periods: int = 45,
    gap_day: int = 20,
    jump: float = 5.0,
) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=periods, freq="B")
    prices = [100.0] * periods
    for i in range(gap_day, periods):
        prices[i] += jump
    return pd.DataFrame({"Close": prices}, index=dates)


def _default_config(**overrides) -> AccountMarketMakingConfig:
    values = dict(
        strikes=(1.0,),
        expiry_days=(30,),
        include_puts=False,
    )
    values.update(overrides)
    return AccountMarketMakingConfig(**values)


def test_no_arrivals_flat_market_reconciles():
    data = _flat_prices()
    result = run_account_market_making(
        price_data=data,
        config=_default_config(arrival_probability=0.0),
    )
    assert result["reconciliation_passed"] is True
    assert abs(result["reconciliation_difference"]) < 1e-6
    assert result["fill_count"] == 0
    assert not result["snapshots"].empty
    assert (result["snapshots"]["equity"] > 0).all()


def test_arrivals_flat_market_reconciles():
    data = _flat_prices()
    result = run_account_market_making(
        price_data=data,
        config=_default_config(arrival_probability=0.8, seed=3),
    )
    assert result["fill_count"] > 0
    assert result["reconciliation_passed"] is True
    assert abs(result["reconciliation_difference"]) < 1e-6
    assert not result["fills"].empty


def test_pre_trade_reduce_breach_blocks_new_fills():
    """Once |delta| would exceed its limit, further option fills are blocked."""
    data = _flat_prices()
    limits = RiskLimits(
        max_gross_exposure=10_000_000.0,
        max_leverage=10.0,
        max_abs_delta=120.0,
        max_abs_gamma=100_000.0,
        max_abs_vega=10_000_000.0,
        max_daily_loss=100_000.0,
        max_drawdown=1.0,
        min_cash_buffer=0.0,
    )
    result = run_account_market_making(
        price_data=data,
        config=_default_config(
            arrival_probability=1.0,
            arrival_side="buy",
            hedge_band_shares=10_000.0,
            risk_limits=limits,
            seed=1,
        ),
    )
    assert result["fill_count"] > 0
    assert result["rejected_fill_count"] > 0
    assert result["halted"] is False
    assert result["reconciliation_passed"] is True


def test_loss_breach_halts_trading_and_still_reconciles():
    """A jump against the short-call inventory trips the daily-loss halt."""
    data = _gap_prices()
    limits = RiskLimits(
        max_gross_exposure=10_000_000.0,
        max_leverage=10.0,
        max_abs_delta=1_000_000.0,
        max_abs_gamma=100_000.0,
        max_abs_vega=10_000_000.0,
        max_daily_loss=400.0,
        max_drawdown=1.0,
        min_cash_buffer=0.0,
    )
    result = run_account_market_making(
        price_data=data,
        config=_default_config(
            arrival_probability=1.0,
            arrival_side="buy",
            risk_limits=limits,
            seed=2,
        ),
    )
    assert result["halted"] is True
    assert result["halt_reasons"]
    assert result["total_pnl"] < 0.0
    assert result["reconciliation_passed"] is True
    assert abs(result["reconciliation_difference"]) < 1e-6


def test_save_account_market_making_result_writes_files(tmp_path):
    data = _flat_prices(periods=35)
    result = run_account_market_making(
        price_data=data,
        config=_default_config(
            arrival_probability=0.5,
            expiry_days=(20,),
        ),
    )
    output = save_account_market_making_result(result, tmp_path)
    assert (output / "snapshots.csv").exists()
    assert (output / "fills.csv").exists()
    assert (output / "summary.json").exists()


def test_default_limits_are_large_enough_for_typical_book():
    limits = default_market_making_limits(200_000.0)
    assert limits.max_daily_loss > 0.0
    assert limits.max_drawdown > 0.0
    assert limits.min_cash_buffer >= 0.0
