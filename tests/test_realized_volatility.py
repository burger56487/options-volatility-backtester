import math

import pandas as pd
import pytest

from src.market_data.realized_volatility import (
    add_realized_volatility_features,
    annualised_realized_volatility,
    calculate_log_returns,
    volatility_summary,
)


def test_log_returns_are_calculated_correctly():
    prices = pd.Series([100.0, 110.0, 121.0])

    returns = calculate_log_returns(prices)

    assert len(returns) == 2
    assert returns.iloc[0] == pytest.approx(math.log(1.10))
    assert returns.iloc[1] == pytest.approx(math.log(1.10))


def test_log_returns_reject_non_positive_prices():
    prices = pd.Series([100.0, 0.0, 101.0])

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        calculate_log_returns(prices)


def test_log_returns_reject_insufficient_price_history():
    prices = pd.Series([100.0])

    with pytest.raises(
        ValueError,
        match="At least two",
    ):
        calculate_log_returns(prices)


def test_annualised_realized_volatility_for_constant_returns():
    returns = pd.Series([0.01] * 10)

    volatility = annualised_realized_volatility(
        log_returns=returns,
        window=5,
        trading_days=252,
    )

    assert volatility.dropna().eq(0.0).all()


def test_realized_volatility_matches_manual_calculation():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])

    volatility = annualised_realized_volatility(
        log_returns=returns,
        window=4,
        trading_days=252,
    )

    expected = returns.std(ddof=1) * math.sqrt(252)

    assert volatility.iloc[-1] == pytest.approx(expected)


def test_invalid_window_raises_error():
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises(
        ValueError,
        match="window must be at least 2",
    ):
        annualised_realized_volatility(
            log_returns=returns,
            window=1,
        )


def test_invalid_min_periods_raises_error():
    returns = pd.Series([0.01, -0.02, 0.03])

    with pytest.raises(
        ValueError,
        match="min_periods must not exceed window",
    ):
        annualised_realized_volatility(
            log_returns=returns,
            window=2,
            min_periods=3,
        )


def test_add_realized_volatility_features():
    dates = pd.date_range(
        "2025-01-01",
        periods=10,
        freq="B",
    )

    data = pd.DataFrame(
        {
            "Close": [
                100.0,
                101.0,
                99.0,
                102.0,
                103.0,
                101.0,
                104.0,
                105.0,
                103.0,
                106.0,
            ]
        },
        index=dates,
    )

    result = add_realized_volatility_features(
        data=data,
        windows=(2, 5),
    )

    assert "log_return" in result.columns
    assert "realised_vol_2d" in result.columns
    assert "realised_vol_5d" in result.columns
    assert result["realised_vol_2d"].notna().sum() > 0
    assert result["realised_vol_5d"].notna().sum() > 0


def test_duplicate_windows_raise_error():
    data = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]}
    )

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        add_realized_volatility_features(
            data=data,
            windows=(2, 2),
        )


def test_volatility_summary_returns_expected_columns():
    data = pd.DataFrame(
        {
            "realised_vol_20d": [0.10, 0.12, 0.15],
            "realised_vol_60d": [0.11, 0.13, 0.14],
        }
    )

    summary = volatility_summary(
        data=data,
        volatility_columns=[
            "realised_vol_20d",
            "realised_vol_60d",
        ],
    )

    assert list(summary.columns) == [
        "count",
        "mean",
        "std",
        "min",
        "median",
        "max",
    ]
    assert list(summary.index) == [
        "realised_vol_20d",
        "realised_vol_60d",
    ]


def test_volatility_summary_rejects_missing_columns():
    data = pd.DataFrame(
        {"realised_vol_20d": [0.10, 0.12]}
    )

    with pytest.raises(
        ValueError,
        match="Missing volatility columns",
    ):
        volatility_summary(
            data=data,
            volatility_columns=[
                "realised_vol_20d",
                "realised_vol_60d",
            ],
        )
