import pandas as pd

from src.backtest.volatility_filter import (
    VolatilityRegimeFilter,
    add_volatility_regime_signal,
)


def _build_price_data(
    periods: int = 420,
    shock_start_index: int | None = None,
) -> pd.DataFrame:
    dates = pd.date_range(
        "2023-01-02",
        periods=periods,
        freq="B",
    )
    prices = [
        100.0
        + 0.02 * index
        + 2.0 * ((index % 7) - 3)
        for index in range(periods)
    ]
    if shock_start_index is not None:
        for index in range(shock_start_index, periods):
            prices[index] = prices[index] * 10.0
    return pd.DataFrame({"Close": prices}, index=dates)


def test_regime_signal_uses_only_past_data():
    full = _build_price_data(shock_start_index=350)
    plain = _build_price_data()

    full_signal = add_volatility_regime_signal(
        full,
        VolatilityRegimeFilter(),
    )
    plain_signal = add_volatility_regime_signal(
        plain,
        VolatilityRegimeFilter(),
    )

    mask = full.index < full.index[350]

    assert full_signal.loc[mask, "volatility_ratio"].equals(
        plain_signal.loc[mask, "volatility_ratio"]
    )
    assert full_signal.loc[mask, "volatility_regime_signal"].equals(
        plain_signal.loc[mask, "volatility_regime_signal"]
    )

    # The test must have teeth: the future shock does change later signals.
    later_mask = full.index > full.index[360]
    assert not full_signal.loc[
        later_mask, "volatility_ratio"
    ].equals(plain_signal.loc[later_mask, "volatility_ratio"])
