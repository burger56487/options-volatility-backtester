"""CSV loaders converting raw tables into schema objects."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schemas import (
    DataType,
    ExerciseStyle,
    OptionQuote,
    OptionType,
    UnderlyingBar,
)


UNDERLYING_REQUIRED_COLUMNS = {
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
}


OPTION_REQUIRED_COLUMNS = {
    "timestamp",
    "underlying_symbol",
    "expiry",
    "strike",
    "option_type",
    "bid",
    "ask",
    "spot",
    "risk_free_rate",
    "dividend_yield",
    "source",
    "data_type",
}


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataset_name: str,
) -> None:
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"{dataset_name} missing required columns: "
            f"{sorted(missing)}"
        )


def load_underlying_csv(
    path: str | Path,
    source: str,
) -> list[UnderlyingBar]:
    """Load a CSV of underlying bars into schema objects."""
    dataframe = pd.read_csv(path)
    require_columns(
        dataframe,
        UNDERLYING_REQUIRED_COLUMNS,
        "underlying",
    )
    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="raise",
    ).dt.date

    bars = []
    for row in dataframe.to_dict("records"):
        bars.append(
            UnderlyingBar(
                trade_date=row["date"],
                symbol=str(row["symbol"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                adjusted_close=float(row["adjusted_close"]),
                volume=float(row["volume"]),
                source=source,
                data_type=DataType.REAL,
            )
        )
    return bars


def load_option_quotes_csv(
    path: str | Path,
) -> list[OptionQuote]:
    """Load a CSV of option quotes into schema objects."""
    dataframe = pd.read_csv(path)
    require_columns(
        dataframe,
        OPTION_REQUIRED_COLUMNS,
        "option quotes",
    )
    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="raise",
    )
    dataframe["expiry"] = pd.to_datetime(
        dataframe["expiry"],
        errors="raise",
    ).dt.date

    quotes = []
    for row in dataframe.to_dict("records"):
        exercise_style = str(
            row.get("exercise_style", "european")
        ).lower()
        quotes.append(
            OptionQuote(
                timestamp=row["timestamp"].to_pydatetime(),
                underlying_symbol=str(row["underlying_symbol"]),
                expiry=row["expiry"],
                strike=float(row["strike"]),
                option_type=OptionType(
                    str(row["option_type"]).lower()
                ),
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                spot=float(row["spot"]),
                risk_free_rate=float(row["risk_free_rate"]),
                dividend_yield=float(row["dividend_yield"]),
                source=str(row["source"]),
                data_type=DataType(
                    str(row["data_type"]).lower()
                ),
                last=(
                    None
                    if pd.isna(row.get("last"))
                    else float(row["last"])
                ),
                volume=(
                    None
                    if pd.isna(row.get("volume"))
                    else int(row["volume"])
                ),
                open_interest=(
                    None
                    if pd.isna(row.get("open_interest"))
                    else int(row["open_interest"])
                ),
                multiplier=int(row.get("multiplier", 100)),
                exercise_style=ExerciseStyle(exercise_style),
            )
        )
    return quotes
