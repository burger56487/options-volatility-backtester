import pandas as pd

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.oos_volatility_filter import (
    chronological_split,
    evaluate_filter_out_of_sample,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
)


def _create_price_data(periods: int = 900) -> pd.DataFrame:
    dates = pd.date_range(
        "2023-01-02",
        periods=periods,
        freq="B",
    )
    prices = [
        100.0
        + 0.03 * index
        + 4.0 * ((index % 9) - 4)
        for index in range(periods)
    ]
    return pd.DataFrame({"Close": prices}, index=dates)


def test_chronological_split_preserves_order():
    data = _create_price_data()
    train, test = chronological_split(data, train_fraction=0.6)
    assert len(train) + len(test) == len(data)
    assert train.index[-1] < test.index[0]


def test_out_of_sample_filter_evaluation_runs():
    data = _create_price_data()
    trade_config = LongStraddleBacktestConfig(
        days_to_expiry=10,
        delta_threshold=1.0,
    )
    rolling_config = RollingBacktestConfig(
        entry_spacing_trading_days=15,
        initial_capital=100_000.0,
    )

    result = evaluate_filter_out_of_sample(
        price_data=data,
        candidate_thresholds=[0.9, 1.0, 1.1],
        trade_config=trade_config,
        rolling_config=rolling_config,
        train_fraction=0.55,
    )

    assert result.selected_threshold in {0.9, 1.0, 1.1}
    assert result.test_result.summary["number_of_trades"] > 0
    assert result.baseline_test_result.summary["number_of_trades"] > 0
    assert len(result.threshold_scores) == 3
