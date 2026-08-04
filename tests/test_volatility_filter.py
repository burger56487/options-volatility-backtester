import pandas as pd
import pytest

from src.backtest.volatility_filter import (
    VolatilityRegimeFilter,
    add_volatility_regime_signal,
    filter_entry_dates,
)


def create_price_data(
    periods: int = 300,
) -> pd.DataFrame:
    dates = pd.date_range(
        "2024-01-02",
        periods=periods,
        freq="B",
    )

    prices = []

    for index in range(periods):
        if index < 250:
            price = 100.0 + 0.05 * index
        else:
            price = (
                112.5
                + 0.05 * index
                + 3.0 * ((index % 5) - 2)
            )

        prices.append(price)

    return pd.DataFrame(
        {"Close": prices},
        index=dates,
    )


def test_add_volatility_regime_signal_creates_columns():
    data = create_price_data()

    result = add_volatility_regime_signal(
        price_data=data,
        regime_filter=VolatilityRegimeFilter(
            short_window=20,
            long_window=252,
            minimum_volatility_ratio=1.0,
        ),
    )

    assert "realised_vol_short" in result.columns
    assert "realised_vol_long" in result.columns
    assert "volatility_ratio" in result.columns
    assert "volatility_regime_signal" in result.columns


def test_high_recent_volatility_can_trigger_signal():
    data = create_price_data()

    result = add_volatility_regime_signal(
        price_data=data,
        regime_filter=VolatilityRegimeFilter(
            short_window=20,
            long_window=252,
            minimum_volatility_ratio=1.10,
        ),
    )

    assert result["volatility_regime_signal"].tail(20).any()


def test_filter_entry_dates_returns_selection_diagnostics():
    data = create_price_data()

    candidate_dates = [
        data.index[260],
        data.index[270],
        data.index[280],
        data.index[290],
    ]

    diagnostics = filter_entry_dates(
        price_data=data,
        candidate_dates=candidate_dates,
        regime_filter=VolatilityRegimeFilter(
            short_window=20,
            long_window=252,
            minimum_volatility_ratio=1.05,
        ),
    )

    assert len(diagnostics) == len(candidate_dates)
    assert "selected" in diagnostics.columns
    assert diagnostics.index.name == "entry_date"


def test_filter_rejects_missing_candidate_dates():
    data = create_price_data()

    with pytest.raises(
        ValueError,
        match="must exist in price_data",
    ):
        filter_entry_dates(
            price_data=data,
            candidate_dates=[
                pd.Timestamp("2030-01-01")
            ],
        )


def test_invalid_filter_configuration_raises_error():
    with pytest.raises(
        ValueError,
        match="short_window must be smaller",
    ):
        VolatilityRegimeFilter(
            short_window=252,
            long_window=20,
        )
