"""Backtest timeline discipline.

The strict daily-frequency protocol used here is:

- information through the previous close updates realised-volatility features;
- a signal generated at the close of day ``t-1`` may only drive a trade that
  is first valued/executed on day ``t``.

The helpers below lag the volatility-regime signal by one day so that a
volatility spike on the candidate entry day itself cannot trigger the entry.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.volatility_filter import (
    VolatilityRegimeFilter,
    add_volatility_regime_signal,
)


def lagged_volatility_regime_signal(
    price_data: pd.DataFrame,
    regime_filter: VolatilityRegimeFilter = (
        VolatilityRegimeFilter()
    ),
) -> pd.DataFrame:
    """Return volatility diagnostics with the signal lagged by one day."""
    featured = add_volatility_regime_signal(
        price_data=price_data,
        regime_filter=regime_filter,
    )
    result = featured.copy()
    result["volatility_ratio_lagged"] = (
        result["volatility_ratio"].shift(1)
    )
    result["volatility_regime_signal_lagged"] = (
        result["volatility_ratio_lagged"]
        >= regime_filter.minimum_volatility_ratio
    )
    result.loc[
        result["volatility_ratio_lagged"].isna(),
        "volatility_regime_signal_lagged",
    ] = False
    return result


def filter_entry_dates_lagged(
    price_data: pd.DataFrame,
    candidate_dates: list[pd.Timestamp],
    regime_filter: VolatilityRegimeFilter = (
        VolatilityRegimeFilter()
    ),
) -> pd.DataFrame:
    """Filter candidate dates using only information up to the prior close."""
    if not candidate_dates:
        raise ValueError("candidate_dates must not be empty.")

    featured = lagged_volatility_regime_signal(
        price_data=price_data,
        regime_filter=regime_filter,
    )
    missing_dates = [
        date
        for date in candidate_dates
        if pd.Timestamp(date) not in featured.index
    ]
    if missing_dates:
        raise ValueError(
            "All candidate_dates must exist in price_data."
        )

    selected = featured.loc[
        [pd.Timestamp(date) for date in candidate_dates],
        [
            "realised_vol_short",
            "realised_vol_long",
            "volatility_ratio_lagged",
            "volatility_regime_signal_lagged",
        ],
    ].copy()
    selected = selected.rename(
        columns={
            "volatility_regime_signal_lagged": "selected",
        }
    )
    selected.index.name = "entry_date"
    return selected


def three_way_split(
    price_data: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train, validation and test windows."""
    train_frac, validation_frac, _ = fractions
    if not (
        0 < train_frac < 1
        and 0 <= validation_frac < 1
        and train_frac + validation_frac < 1
    ):
        raise ValueError(
            "fractions must satisfy 0 < train < 1 and "
            "0 <= validation and train + validation < 1."
        )

    total = len(price_data)
    train_end = int(total * train_frac)
    validation_end = int(total * (train_frac + validation_frac))
    return (
        price_data.iloc[:train_end].copy(),
        price_data.iloc[train_end:validation_end].copy(),
        price_data.iloc[validation_end:].copy(),
    )
