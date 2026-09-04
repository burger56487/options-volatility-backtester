import pandas as pd

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
    run_long_straddle_backtest,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
    run_rolling_long_straddle_backtest,
)


def _create_price_data(periods: int = 300) -> pd.DataFrame:
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
    return pd.DataFrame({"Close": prices}, index=dates)


def _metadata_keys(summary: dict) -> set[str]:
    return {
        "underlying_data_type",
        "option_data_type",
        "execution_type",
        "strategy_version",
        "git_commit",
    }


def test_single_trade_summary_contains_experiment_metadata():
    data = _create_price_data()
    result = run_long_straddle_backtest(
        price_data=data,
        entry_date=data.index[270],
        config=LongStraddleBacktestConfig(days_to_expiry=10),
    )
    assert _metadata_keys(result.summary).issubset(result.summary)
    assert result.summary["underlying_data_type"] == "real"
    assert result.summary["option_data_type"] == "synthetic"
    assert result.summary["execution_type"] == "simulated"


def test_rolling_summary_contains_experiment_metadata():
    data = _create_price_data()
    result = run_rolling_long_straddle_backtest(
        price_data=data,
        trade_config=LongStraddleBacktestConfig(
            days_to_expiry=10,
        ),
        rolling_config=RollingBacktestConfig(
            entry_spacing_trading_days=15,
        ),
    )
    assert _metadata_keys(result.summary).issubset(result.summary)
    assert result.summary["option_data_type"] == "synthetic"
