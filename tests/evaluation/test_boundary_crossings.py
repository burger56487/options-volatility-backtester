"""Boundary-crossing trade handling in the strict evaluation runner."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.evaluation.runner import (
    boundary_crossing_trades,
    label_trades_by_split,
)
from src.evaluation.splits import (
    DateSplit,
    TrainValidationTestSplit,
)


def _split() -> TrainValidationTestSplit:
    return TrainValidationTestSplit(
        train=DateSplit(
            name="train",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            purpose="parameter_estimation",
        ),
        validation=DateSplit(
            name="validation",
            start_date=date(2024, 7, 1),
            end_date=date(2024, 9, 30),
            purpose="model_selection",
        ),
        test=DateSplit(
            name="test",
            start_date=date(2024, 10, 1),
            end_date=date(2024, 12, 31),
            purpose="final_evaluation",
        ),
    )


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_date": pd.Timestamp("2024-03-01"),
                "expiry_date": pd.Timestamp("2024-04-15"),
            },
            {
                "entry_date": pd.Timestamp("2024-06-15"),
                "expiry_date": pd.Timestamp("2024-08-01"),
            },
            {
                "entry_date": pd.Timestamp("2024-08-01"),
                "expiry_date": pd.Timestamp("2024-08-30"),
            },
        ]
    )


def test_boundary_crossing_preserves_original_index():
    split = _split()
    labelled = label_trades_by_split(_trades(), split)
    train_crossing = boundary_crossing_trades(labelled, split.train)
    assert list(train_crossing.index) == [1]

    train_all = labelled[labelled["split_name"] == "train"]
    clean = train_all[~train_all.index.isin(train_crossing.index)]
    assert list(clean["entry_date"]) == [pd.Timestamp("2024-03-01")]


def test_validation_crossing_is_separated_from_clean_validation():
    split = _split()
    labelled = label_trades_by_split(_trades(), split)
    validation_crossing = boundary_crossing_trades(
        labelled, split.validation
    )
    assert validation_crossing.empty
