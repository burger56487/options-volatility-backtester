import pandas as pd
import pytest

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.sensitivity_analysis import (
    create_sensitivity_pivot,
    run_hedging_sensitivity_analysis,
)


def create_price_data(
    periods: int = 330,
) -> pd.DataFrame:
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


def test_sensitivity_analysis_returns_all_grid_combinations():
    data = create_price_data()
    entry_date = data.index[260]

    results = run_hedging_sensitivity_analysis(
        price_data=data,
        entry_date=entry_date,
        base_config=LongStraddleBacktestConfig(
            days_to_expiry=10,
        ),
        delta_thresholds=(0.0, 5.0),
        slippage_bps_values=(1.0, 5.0),
    )

    assert len(results) == 4
    assert set(results["delta_threshold"]) == {0.0, 5.0}
    assert set(results["slippage_bps"]) == {1.0, 5.0}
    assert "final_pnl" in results.columns
    assert "cumulative_hedge_costs" in results.columns


def test_higher_slippage_increases_costs_when_hedging_occurs():
    data = create_price_data()
    entry_date = data.index[260]

    results = run_hedging_sensitivity_analysis(
        price_data=data,
        entry_date=entry_date,
        base_config=LongStraddleBacktestConfig(
            days_to_expiry=10,
            delta_threshold=0.0,
        ),
        delta_thresholds=(0.0,),
        slippage_bps_values=(0.0, 10.0),
    )

    zero_slippage_cost = results.loc[
        results["slippage_bps"] == 0.0,
        "cumulative_hedge_costs",
    ].iloc[0]

    high_slippage_cost = results.loc[
        results["slippage_bps"] == 10.0,
        "cumulative_hedge_costs",
    ].iloc[0]

    assert high_slippage_cost > zero_slippage_cost


def test_higher_threshold_does_not_increase_number_of_hedge_trades():
    data = create_price_data()
    entry_date = data.index[260]

    results = run_hedging_sensitivity_analysis(
        price_data=data,
        entry_date=entry_date,
        base_config=LongStraddleBacktestConfig(
            days_to_expiry=10,
        ),
        delta_thresholds=(0.0, 1000.0),
        slippage_bps_values=(1.0,),
    )

    low_threshold_trades = results.loc[
        results["delta_threshold"] == 0.0,
        "number_of_hedge_trades",
    ].iloc[0]

    high_threshold_trades = results.loc[
        results["delta_threshold"] == 1000.0,
        "number_of_hedge_trades",
    ].iloc[0]

    assert high_threshold_trades <= low_threshold_trades


def test_create_sensitivity_pivot():
    results = pd.DataFrame(
        {
            "delta_threshold": [0.0, 0.0, 5.0, 5.0],
            "slippage_bps": [1.0, 5.0, 1.0, 5.0],
            "final_pnl": [10.0, 8.0, 9.0, 7.0],
        }
    )

    pivot = create_sensitivity_pivot(
        results=results,
        value_column="final_pnl",
    )

    assert pivot.shape == (2, 2)
    assert pivot.loc[0.0, 1.0] == pytest.approx(10.0)
    assert pivot.loc[5.0, 5.0] == pytest.approx(7.0)


def test_sensitivity_analysis_rejects_negative_threshold():
    data = create_price_data()
    entry_date = data.index[260]

    with pytest.raises(
        ValueError,
        match="delta_thresholds must be non-negative",
    ):
        run_hedging_sensitivity_analysis(
            price_data=data,
            entry_date=entry_date,
            base_config=LongStraddleBacktestConfig(
                days_to_expiry=10,
            ),
            delta_thresholds=(-1.0,),
        )


def test_pivot_rejects_missing_value_column():
    results = pd.DataFrame(
        {
            "delta_threshold": [0.0],
            "slippage_bps": [1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        create_sensitivity_pivot(
            results=results,
            value_column="final_pnl",
        )
