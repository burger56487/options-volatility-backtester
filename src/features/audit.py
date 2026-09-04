"""Detect feature frames that use same-day or future observations."""

from __future__ import annotations

import pandas as pd


def audit_feature_timing(
    dataframe: pd.DataFrame,
    signal_column: str = "signal_date",
    observation_column: str = "observation_end_date",
) -> pd.DataFrame:
    """Return rows whose observation end is not strictly before signal."""
    required = {signal_column, observation_column}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(
            f"Feature frame missing timing columns: {sorted(missing)}"
        )
    result = dataframe.copy()
    result[signal_column] = pd.to_datetime(result[signal_column])
    result[observation_column] = pd.to_datetime(
        result[observation_column]
    )
    return result[
        result[observation_column] >= result[signal_column]
    ].copy()


def assert_no_feature_lookahead(
    dataframe: pd.DataFrame,
) -> None:
    violations = audit_feature_timing(dataframe)
    if not violations.empty:
        columns = [
            column
            for column in [
                "date",
                "signal_date",
                "observation_end_date",
            ]
            if column in violations.columns
        ]
        raise ValueError(
            "Feature look-ahead detected:\n"
            f"{violations[columns].head(10).to_string(index=False)}"
        )
