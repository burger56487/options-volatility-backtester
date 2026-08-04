import pandas as pd
import pytest

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
    run_rolling_long_straddle_backtest,
    save_rolling_backtest_result,
)


def create_price_data(
    periods: int = 450,
) -> pd.DataFrame:
    """Create deterministic price data with sufficient history."""
    dates = pd.date_range(
        "2023-01-02",
        periods=periods,
        freq="B",
    )

    prices = [
        100.0
        + 0.04 * index
        + 3.0 * ((index % 11) - 5)
        for index in range(periods)
    ]

    return pd.DataFrame(
        {"Close": prices},
        index=dates,
    )


def test_rolling_backtest_returns_multiple_trades():
    data = create_price_data()

    result = run_rolling_long_straddle_backtest(
        price_data=data,
        trade_config=LongStraddleBacktestConfig(
            days_to_expiry=10,
            delta_threshold=1.0,
        ),
        rolling_config=RollingBacktestConfig(
            entry_spacing_trading_days=15,
            initial_capital=100_000.0,
        ),
    )

    assert len(result.trade_results) > 1
    assert not result.equity_curve.empty
    assert "final_pnl" in result.trade_results.columns
    assert "equity" in result.equity_curve.columns
    assert result.summary["number_of_trades"] == len(
        result.trade_results
    )


def test_capital_after_trade_matches_cumulative_pnl():
    data = create_price_data()

    initial_capital = 50_000.0

    result = run_rolling_long_straddle_backtest(
        price_data=data,
        trade_config=LongStraddleBacktestConfig(
            days_to_expiry=10,
        ),
        rolling_config=RollingBacktestConfig(
            entry_spacing_trading_days=20,
            initial_capital=initial_capital,
        ),
    )

    expected_final_capital = (
        initial_capital
        + result.trade_results["final_pnl"].sum()
    )

    assert result.summary["final_capital"] == pytest.approx(
        expected_final_capital
    )


def test_rolling_backtest_rejects_insufficient_data():
    dates = pd.date_range(
        "2025-01-01",
        periods=50,
        freq="B",
    )

    data = pd.DataFrame(
        {"Close": [100.0 + index for index in range(50)]},
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="Insufficient data",
    ):
        run_rolling_long_straddle_backtest(
            price_data=data,
            trade_config=LongStraddleBacktestConfig(
                days_to_expiry=10,
            ),
        )


def test_save_rolling_result_writes_files(tmp_path):
    data = create_price_data()

    result = run_rolling_long_straddle_backtest(
        price_data=data,
        trade_config=LongStraddleBacktestConfig(
            days_to_expiry=10,
        ),
        rolling_config=RollingBacktestConfig(
            entry_spacing_trading_days=20,
        ),
    )

    save_rolling_backtest_result(
        result=result,
        output_directory=tmp_path,
        prefix="test_rolling",
    )

    assert (
        tmp_path / "test_rolling_trade_results.csv"
    ).exists()

    assert (
        tmp_path / "test_rolling_equity_curve.csv"
    ).exists()

    assert (
        tmp_path / "test_rolling_summary.json"
    ).exists()
