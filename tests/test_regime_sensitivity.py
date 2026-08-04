import pandas as pd
import pytest

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.regime_sensitivity import (
    run_regime_threshold_sensitivity,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
)


def create_price_data(
    periods: int = 450,
) -> pd.DataFrame:
    dates = pd.date_range(
        "2023-01-02",
        periods=periods,
        freq="B",
    )

    prices = []

    for index in range(periods):
        base = 100.0 + 0.05 * index

        if index % 50 < 25:
            shock = 1.0 * ((index % 5) - 2)
        else:
            shock = 4.0 * ((index % 7) - 3)

        prices.append(base + shock)

    return pd.DataFrame(
        {"Close": prices},
        index=dates,
    )


def test_regime_sensitivity_returns_one_row_per_threshold():
    data = create_price_data()

    results = run_regime_threshold_sensitivity(
        price_data=data,
        trade_config=LongStraddleBacktestConfig(
            days_to_expiry=10,
        ),
        rolling_config=RollingBacktestConfig(
            entry_spacing_trading_days=20,
        ),
        thresholds=(0.90, 1.10),
    )

    assert len(results) == 2
    assert list(results["volatility_ratio_threshold"]) == [
        0.90,
        1.10,
    ]
    assert "total_pnl" in results.columns
    assert "number_of_trades" in results.columns
    assert "sharpe_like_ratio" in results.columns


def test_higher_threshold_does_not_select_more_dates():
    data = create_price_data()

    results = run_regime_threshold_sensitivity(
        price_data=data,
        trade_config=LongStraddleBacktestConfig(
            days_to_expiry=10,
        ),
        rolling_config=RollingBacktestConfig(
            entry_spacing_trading_days=20,
        ),
        thresholds=(0.90, 1.10),
    )

    low_threshold_trades = results.loc[
        results["volatility_ratio_threshold"] == 0.90,
        "number_of_trades",
    ].iloc[0]

    high_threshold_trades = results.loc[
        results["volatility_ratio_threshold"] == 1.10,
        "number_of_trades",
    ].iloc[0]

    assert high_threshold_trades <= low_threshold_trades



def test_duplicate_thresholds_raise_error():
    data = create_price_data()

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        run_regime_threshold_sensitivity(
            price_data=data,
            trade_config=LongStraddleBacktestConfig(
                days_to_expiry=10,
            ),
            rolling_config=RollingBacktestConfig(
                entry_spacing_trading_days=20,
            ),
            thresholds=(1.0, 1.0),
        )


def test_non_positive_threshold_raises_error():
    data = create_price_data()

    with pytest.raises(
        ValueError,
        match="thresholds must be positive",
    ):
        run_regime_threshold_sensitivity(
            price_data=data,
            trade_config=LongStraddleBacktestConfig(
                days_to_expiry=10,
            ),
            rolling_config=RollingBacktestConfig(
                entry_spacing_trading_days=20,
            ),
            thresholds=(0.0,),
        )
