from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


REQUIRED_PRICE_COLUMNS = {
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
}


@dataclass(frozen=True)
class DataQualityReport:
    """Summary statistics for underlying-price data quality."""

    observations: int
    start_date: str
    end_date: str
    duplicate_dates: int
    missing_values: int
    non_positive_prices: int
    invalid_high_low_rows: int
    negative_volume_rows: int


def _flatten_yfinance_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flatten yfinance MultiIndex columns for single-ticker downloads.

    Recent yfinance versions may return columns such as:
    ('Close', 'SPY') instead of 'Close'.
    """
    result = data.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)

    return result


def download_price_data(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download daily adjusted OHLCV data from yfinance.

    Prices are adjusted for splits and dividends through auto_adjust=True.
    """
    if not ticker.strip():
        raise ValueError("ticker must not be empty.")

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
    )

    if data.empty:
        raise ValueError(
            "No price data returned. Check ticker and date range."
        )

    data = _flatten_yfinance_columns(data)
    data.index = pd.to_datetime(data.index)
    data.index.name = "Date"

    return data


def validate_price_data(
    data: pd.DataFrame,
) -> DataQualityReport:
    """
    Validate daily OHLCV data and return a data-quality summary.

    This function does not modify the input data.
    """
    if data.empty:
        raise ValueError("Price data is empty.")

    missing_columns = REQUIRED_PRICE_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Price data index must be a DatetimeIndex.")

    duplicate_dates = int(data.index.duplicated().sum())
    missing_values = int(data.isna().sum().sum())

    price_columns = ["Open", "High", "Low", "Close"]

    non_positive_prices = int(
        (data[price_columns] <= 0).sum().sum()
    )

    invalid_high_low_rows = int(
        (
            (data["High"] < data["Low"])
            | (data["High"] < data["Open"])
            | (data["High"] < data["Close"])
            | (data["Low"] > data["Open"])
            | (data["Low"] > data["Close"])
        ).sum()
    )

    negative_volume_rows = int((data["Volume"] < 0).sum())

    return DataQualityReport(
        observations=len(data),
        start_date=str(data.index.min().date()),
        end_date=str(data.index.max().date()),
        duplicate_dates=duplicate_dates,
        missing_values=missing_values,
        non_positive_prices=non_positive_prices,
        invalid_high_low_rows=invalid_high_low_rows,
        negative_volume_rows=negative_volume_rows,
    )


def clean_price_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort data, remove duplicate dates, and enforce a clean Date index.

    Raises ValueError if missing values or invalid OHLCV observations remain.
    """
    report = validate_price_data(data)

    if report.missing_values > 0:
        raise ValueError(
            "Price data contains missing values."
        )

    if report.non_positive_prices > 0:
        raise ValueError(
            "Price data contains non-positive prices."
        )

    if report.invalid_high_low_rows > 0:
        raise ValueError(
            "Price data contains invalid OHLC relationships."
        )

    if report.negative_volume_rows > 0:
        raise ValueError(
            "Price data contains negative volume."
        )

    result = data.copy()
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()

    return result


def add_return_features(
    data: pd.DataFrame,
    price_column: str = "Close",
) -> pd.DataFrame:
    """
    Add simple and logarithmic return features.

    New columns:
    - simple_return;
    - log_return.
    """
    if price_column not in data.columns:
        raise ValueError(
            f"Price column '{price_column}' is not available."
        )

    if (data[price_column] <= 0).any():
        raise ValueError(
            "Prices must be positive to calculate returns."
        )

    result = data.copy()

    result["simple_return"] = (
        result[price_column].pct_change()
    )

    result["log_return"] = np.log(
        result[price_column] / result[price_column].shift(1)
    )

    return result


def save_price_data(
    data: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save price data as CSV, creating parent directories if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    result = data.copy()
    result.index.name = "Date"

    result.to_csv(path, index=True)



def load_price_data(
    input_path: str | Path,
) -> pd.DataFrame:
    """Load a previously saved CSV price dataset."""
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Price data file not found: {path}"
        )

    data = pd.read_csv(
        path,
        index_col="Date",
        parse_dates=True,
    )

    return data
