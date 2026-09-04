import pandas as pd

from src.backtest.timeline import (
    filter_entry_dates_lagged,
    three_way_split,
)
from src.backtest.volatility_filter import (
    VolatilityRegimeFilter,
    filter_entry_dates,
)


def _spiky_price_data(periods: int = 400) -> pd.DataFrame:
    dates = pd.date_range(
        "2023-01-02",
        periods=periods,
        freq="B",
    )
    prices = [
        100.0 + 0.02 * index for index in range(periods)
    ]
    prices[300] = prices[300] * 1.8  # single-day volatility spike
    prices[301] = prices[300] * 0.55
    return pd.DataFrame({"Close": prices}, index=dates)


def test_lagged_filter_ignores_same_day_spike():
    data = _spiky_price_data()
    regime = VolatilityRegimeFilter(minimum_volatility_ratio=1.05)
    candidate = pd.Timestamp(data.index[300])

    same_day = filter_entry_dates(
        price_data=data,
        candidate_dates=[candidate],
        regime_filter=regime,
    )
    lagged = filter_entry_dates_lagged(
        price_data=data,
        candidate_dates=[candidate],
        regime_filter=regime,
    )

    # With data through the close of day 300 available, the spike is visible.
    assert bool(same_day.loc[candidate, "selected"])
    # A strict timeline uses only data through the close of day 299.
    assert not bool(lagged.loc[candidate, "selected"])


def test_three_way_split_preserves_order():
    dates = pd.date_range("2023-01-02", periods=400, freq="B")
    data = pd.DataFrame({"Close": range(400)}, index=dates)
    train, validation, test = three_way_split(
        data,
        fractions=(0.5, 0.25, 0.25),
    )
    assert train.index[-1] < validation.index[0]
    assert validation.index[-1] < test.index[0]
    assert len(train) + len(validation) + len(test) == 400
