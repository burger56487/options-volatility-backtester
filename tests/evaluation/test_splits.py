from datetime import date

import pytest

from src.evaluation.splits import DateSplit, TrainValidationTestSplit
from src.evaluation.walk_forward import (
    generate_expanding_window_folds,
)


def _make_split() -> TrainValidationTestSplit:
    return TrainValidationTestSplit(
        train=DateSplit(
            "train",
            date(2021, 1, 1),
            date(2023, 12, 31),
            "parameter_estimation",
        ),
        validation=DateSplit(
            "validation",
            date(2024, 1, 1),
            date(2024, 12, 31),
            "model_selection",
        ),
        test=DateSplit(
            "test",
            date(2025, 1, 1),
            date(2025, 12, 31),
            "final_evaluation",
        ),
    )


def test_non_overlapping_split_is_valid():
    _make_split().validate()


def test_overlapping_split_is_rejected():
    split = TrainValidationTestSplit(
        train=DateSplit(
            "train",
            date(2021, 1, 1),
            date(2024, 1, 1),
            "parameter_estimation",
        ),
        validation=DateSplit(
            "validation",
            date(2024, 1, 1),
            date(2024, 12, 31),
            "model_selection",
        ),
        test=DateSplit(
            "test",
            date(2025, 1, 1),
            date(2025, 12, 31),
            "final_evaluation",
        ),
    )
    with pytest.raises(ValueError):
        split.validate()


def test_expanding_window_folds_generated():
    dates = __import__("pandas").bdate_range(
        "2021-01-01",
        periods=800,
    )
    folds = generate_expanding_window_folds(
        dates=dates,
        minimum_train_observations=504,
        validation_observations=126,
        test_observations=126,
    )
    assert len(folds) >= 1
    assert folds[0].train_end < folds[0].validation_start
    assert folds[0].validation_end < folds[0].test_start
