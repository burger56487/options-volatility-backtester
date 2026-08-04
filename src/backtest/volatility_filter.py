from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.market_data.realized_volatility import (
    add_realized_volatility_features,
)


@dataclass(frozen=True)
class VolatilityRegimeFilter:
    """
    Volatility-regime filter based on short-/long-horizon realised volatility.

    A date passes when:

        realised_vol_short / realised_vol_long >= minimum_volatility_ratio

    For example, a ratio of 1.10 requires short-horizon realised volatility
    to be at least 10% above long-horizon realised volatility.
    """

    short_window: int = 20
    long_window: int = 252
    minimum_volatility_ratio: float = 1.10

    def __post_init__(self) -> None:
        if self.short_window < 2:
            raise ValueError(
                "short_window must be at least 2."
            )

        if self.long_window < 2:
            raise ValueError(
                "long_window must be at least 2."
            )

        if self.short_window >= self.long_window:
            raise ValueError(
                "short_window must be smaller than long_window."
            )

        if self.minimum_volatility_ratio <= 0:
            raise ValueError(
                "minimum_volatility_ratio must be positive."
            )


def add_volatility_regime_signal(
    price_data: pd.DataFrame,
    regime_filter: VolatilityRegimeFilter = (
        VolatilityRegimeFilter()
    ),
) -> pd.DataFrame:
    """
    Add short/long realised-volatility ratio and a boolean entry signal.

    New columns:
    - realised_vol_short;
    - realised_vol_long;
    - volatility_ratio;
    - volatility_regime_signal.
    """
    featured = add_realized_volatility_features(
        data=price_data,
        price_column="Close",
        windows=(
            regime_filter.short_window,
            regime_filter.long_window,
        ),
    )

    short_column = (
        f"realised_vol_{regime_filter.short_window}d"
    )
    long_column = (
        f"realised_vol_{regime_filter.long_window}d"
    )

    result = featured.copy()

    result["realised_vol_short"] = result[short_column]
    result["realised_vol_long"] = result[long_column]

    result["volatility_ratio"] = (
        result["realised_vol_short"]
        / result["realised_vol_long"]
    )

    result["volatility_regime_signal"] = (
        result["volatility_ratio"]
        >= regime_filter.minimum_volatility_ratio
    )

    result.loc[
        result[
            [
                "realised_vol_short",
                "realised_vol_long",
            ]
        ].isna().any(axis=1),
        "volatility_regime_signal",
    ] = False

    return result


def filter_entry_dates(
    price_data: pd.DataFrame,
    candidate_dates: list[pd.Timestamp],
    regime_filter: VolatilityRegimeFilter = (
        VolatilityRegimeFilter()
    ),
) -> pd.DataFrame:
    """
    Return candidate entry dates together with volatility-regime diagnostics.

    Only candidate dates are returned. The `selected` column indicates whether
    each candidate passes the volatility filter.
    """
    if not candidate_dates:
        raise ValueError(
            "candidate_dates must not be empty."
        )

    featured = add_volatility_regime_signal(
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
            "volatility_ratio",
            "volatility_regime_signal",
        ],
    ].copy()

    selected = selected.rename(
        columns={
            "volatility_regime_signal": "selected",
        }
    )

    selected.index.name = "entry_date"

    return selected
