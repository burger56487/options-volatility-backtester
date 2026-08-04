import pandas as pd
import pytest

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
    run_long_straddle_backtest,
    save_backtest_result,
)


def create_price_data(
    periods: int = 330,
) -> pd.DataFrame:
    """
    Create deterministic synthetic daily price data long enough
    to support a 252-day realised-volatility window.
    """
    dates = pd.date_range(
        "2024-01-02",
        periods=periods,
        freq="B",
    )

    prices = [
        100.0 * (1.0 + 0.0005 * index)
        + 2.0 * ((index % 7) - 3)
        for index in range(periods)
    ]

    return pd.DataFrame(
        {"Close": prices},
        index=dates,
    )


def test_backtest_returns_equity_curve_and_summary():
    data = create_price_data()
    entry_date = data.index[260]

    result = run_long_straddle_backtest(
        price_data=data,
        entry_date=entry_date,
        config=LongStraddleBacktestConfig(
            days_to_expiry=20,
            delta_threshold=0.0,
        ),
    )

    assert not result.equity_curve.empty
    assert "total_pnl" in result.equity_curve.columns
    assert "option_delta" in result.equity_curve.columns
    assert "hedge_position" in result.equity_curve.columns
    assert result.summary["entry_date"] == str(entry_date.date())
    assert result.summary["number_of_hedge_trades"] >= 1


def test_backtest_initial_pnl_is_negative_due_to_spreads_and_hedging_costs():
    data = create_price_data()
    entry_date = data.index[260]

    result = run_long_straddle_backtest(
        price_data=data,
        entry_date=entry_date,
        config=LongStraddleBacktestConfig(
            days_to_expiry=20,
            delta_threshold=0.0,
            commission_per_share=0.005,
            underlying_slippage_bps=1.0,
        ),
    )

    initial_pnl = result.equity_curve["total_pnl"].iloc[0]

    assert initial_pnl < 0


def test_backtest_post_hedge_delta_is_small_with_fractional_shares():
    data = create_price_data()
    entry_date = data.index[260]

    result = run_long_straddle_backtest(
        price_data=data,
        entry_date=entry_date,
        config=LongStraddleBacktestConfig(
            days_to_expiry=20,
            delta_threshold=0.0,
            allow_fractional_shares=True,
        ),
    )

    post_hedge_delta = result.equity_curve[
        "post_hedge_delta"
    ]

    assert post_hedge_delta.abs().max() < 1e-8


def test_backtest_requires_sufficient_history():
    data = create_price_data(periods=100)

    with pytest.raises(
        ValueError,
        match="Insufficient price history",
    ):
        run_long_straddle_backtest(
            price_data=data,
            entry_date=data.index[-1],
        )


def test_backtest_rejects_entry_date_not_in_data():
    data = create_price_data()

    with pytest.raises(
        ValueError,
        match="entry_date must exist",
    ):
        run_long_straddle_backtest(
            price_data=data,
            entry_date=pd.Timestamp("2030-01-01"),
        )


def test_save_backtest_result_writes_artifacts(tmp_path):
    data = create_price_data()
    entry_date = data.index[260]

    result = run_long_straddle_backtest(
        price_data=data,
        entry_date=entry_date,
        config=LongStraddleBacktestConfig(
            days_to_expiry=10,
        ),
    )

    save_backtest_result(
        result=result,
        output_directory=tmp_path,
        prefix="test_trade",
    )

    assert (
        tmp_path / "test_trade_equity_curve.csv"
    ).exists()

    assert (
        tmp_path / "test_trade_hedge_trades.csv"
    ).exists()

    assert (
        tmp_path / "test_trade_summary.json"
    ).exists()
