from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd


def calculate_log_returns(
    prices: pd.Series,
) -> pd.Series:
    """
    Calculate logarithmic returns from a strictly positive price series.

    The first observation is removed because no prior price is available.
    """
    if prices.empty:
        raise ValueError("prices must not be empty.")

    clean_prices = prices.dropna()

    if len(clean_prices) < 2:
        raise ValueError(
            "At least two non-null prices are required."
        )

    if (clean_prices <= 0).any():
        raise ValueError("prices must be strictly positive.")

    returns = np.log(clean_prices / clean_prices.shift(1))

    return returns.dropna()


def annualised_realized_volatility(
    log_returns: pd.Series,
    window: int,
    trading_days: int = 252,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Calculate rolling annualised realised volatility.

    Volatility is estimated as the rolling standard deviation of log returns,
    multiplied by the square root of the annual trading-day count.
    """
    if log_returns.empty:
        raise ValueError("log_returns must not be empty.")

    if window < 2:
        raise ValueError("window must be at least 2.")

    if trading_days < 1:
        raise ValueError("trading_days must be at least 1.")

    if min_periods is None:
        min_periods = window

    if min_periods < 2:
        raise ValueError("min_periods must be at least 2.")

    if min_periods > window:
        raise ValueError(
            "min_periods must not exceed window."
        )

    clean_returns = log_returns.dropna()

    return (
        clean_returns.rolling(
            window=window,
            min_periods=min_periods,
        ).std(ddof=1)
        * sqrt(trading_days)
    )


def add_realized_volatility_features(
    data: pd.DataFrame,
    price_column: str = "Close",
    windows: tuple[int, ...] = (20, 60, 252),
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Add log returns and rolling realised-volatility features to price data.

    New columns include:
    - log_return;
    - realised_vol_{window}d for every requested window.
    """
    if data.empty:
        raise ValueError("price data must not be empty.")

    if price_column not in data.columns:
        raise ValueError(
            f"Price column '{price_column}' is not available."
        )

    if not windows:
        raise ValueError("windows must not be empty.")

    if len(set(windows)) != len(windows):
        raise ValueError("windows must not contain duplicates.")

    if any(window < 2 for window in windows):
        raise ValueError(
            "all volatility windows must be at least 2."
        )

    if (data[price_column].dropna() <= 0).any():
        raise ValueError(
            "prices must be strictly positive."
        )

    result = data.copy()
    log_returns = calculate_log_returns(result[price_column])

    result["log_return"] = log_returns

    for window in windows:
        column_name = f"realised_vol_{window}d"

        result[column_name] = annualised_realized_volatility(
            log_returns=log_returns,
            window=window,
            trading_days=trading_days,
        )

    return result


def volatility_summary(
    data: pd.DataFrame,
    volatility_columns: list[str],
) -> pd.DataFrame:
    """
    Create a summary table for realised-volatility features.

    Returns count, mean, standard deviation, minimum, median, and maximum.
    """
    missing_columns = (
        set(volatility_columns) - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing volatility columns: {sorted(missing_columns)}"
        )

    summary = data[volatility_columns].describe().T

    return summary[
        ["count", "mean", "std", "min", "50%", "max"]
    ].rename(columns={"50%": "median"})
