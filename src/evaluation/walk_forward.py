"""Expanding-window walk-forward folds."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_expanding_window_folds(
    dates: pd.Series,
    minimum_train_observations: int,
    validation_observations: int,
    test_observations: int,
    step_observations: int | None = None,
) -> list[WalkForwardFold]:
    """Generate expanding-window folds over a trading-observation axis."""
    unique_dates = (
        pd.Series(pd.to_datetime(dates).dropna().unique())
        .sort_values()
        .reset_index(drop=True)
    )
    if step_observations is None:
        step_observations = test_observations

    required = (
        minimum_train_observations
        + validation_observations
        + test_observations
    )
    if len(unique_dates) < required:
        raise ValueError("Insufficient dates for expanding-window folds.")

    folds = []
    test_end_index = required - 1
    fold_id = 1
    while test_end_index < len(unique_dates):
        train_end_index = (
            test_end_index
            - validation_observations
            - test_observations
        )
        validation_start_index = train_end_index + 1
        validation_end_index = test_end_index - test_observations
        test_start_index = validation_end_index + 1
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=unique_dates.iloc[0],
                train_end=unique_dates.iloc[train_end_index],
                validation_start=unique_dates.iloc[
                    validation_start_index
                ],
                validation_end=unique_dates.iloc[
                    validation_end_index
                ],
                test_start=unique_dates.iloc[test_start_index],
                test_end=unique_dates.iloc[test_end_index],
            )
        )
        fold_id += 1
        test_end_index += step_observations
    return folds
