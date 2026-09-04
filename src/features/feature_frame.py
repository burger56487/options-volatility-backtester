"""Volatility feature frames whose availability date is explicit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpecification:
    name: str
    lookback_days: int
    publication_delay_days: int = 1


def build_realised_volatility_features(
    prices: pd.DataFrame,
    annualisation_factor: int = 252,
) -> pd.DataFrame:
    """Build realised-volatility features available one day after observed."""
    dataframe = prices.copy()
    dataframe = dataframe.sort_values("date").reset_index(drop=True)
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe["log_return"] = np.log(
        dataframe["adjusted_close"].astype(float)
    ).diff()

    for window in [20, 60, 252]:
        raw_column = f"rv_{window}_observed"
        available_column = f"rv_{window}"
        dataframe[raw_column] = (
            dataframe["log_return"]
            .rolling(window=window, min_periods=window)
            .std()
            * annualisation_factor**0.5
        )
        # Observed at the close of day t, usable from day t+1 onwards.
        dataframe[available_column] = dataframe[raw_column].shift(1)

    dataframe["vol_ratio_20_252"] = (
        dataframe["rv_20"] / dataframe["rv_252"]
    )
    dataframe["observation_end_date"] = dataframe["date"].shift(1)
    dataframe["signal_date"] = dataframe["date"]
    dataframe.attrs["feature_timing"] = (
        "Features on signal_date use observations ending no later "
        "than the previous trading day."
    )
    return dataframe
