"""Signal helpers with explicit no-future-data discipline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol_forecast_from_past(
    returns: pd.Series,
    window: int = 20,
    annualisation: int = 252,
) -> pd.Series:
    """Forecast realised vol for the next period using only trailing data.

    The value at date t uses returns strictly before t (shift(1)) so it can
    never touch the period it is forecasting.
    """
    trailing = returns.rolling(window, min_periods=window).std().shift(1)
    return trailing * np.sqrt(annualisation)


def vrp_signal(
    implied_vol: pd.Series,
    realized_forecast: pd.Series,
    threshold: float = 1.0,
) -> pd.Series:
    """Long-vol premium signal: implied vol above a trailing forecast."""
    aligned = pd.concat(
        [implied_vol, realized_forecast],
        axis=1,
    ).dropna()
    return aligned.iloc[:, 0] / aligned.iloc[:, 1] >= threshold
