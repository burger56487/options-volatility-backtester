"""Non-overlapping chronological dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class DateSplit:
    name: str
    start_date: date
    end_date: date
    purpose: str

    def contains(self, timestamp) -> bool:
        current_date = pd.Timestamp(timestamp).date()
        return self.start_date <= current_date <= self.end_date


@dataclass(frozen=True)
class TrainValidationTestSplit:
    train: DateSplit
    validation: DateSplit
    test: DateSplit

    def validate(self) -> None:
        if not (
            self.train.end_date < self.validation.start_date
            and self.validation.end_date < self.test.start_date
        ):
            raise ValueError(
                "Train, validation and test must be chronological "
                "and non-overlapping."
            )

    def locate(self, timestamp) -> str | None:
        for split in [self.train, self.validation, self.test]:
            if split.contains(timestamp):
                return split.name
        return None

    def as_list(self) -> list[DateSplit]:
        return [self.train, self.validation, self.test]
