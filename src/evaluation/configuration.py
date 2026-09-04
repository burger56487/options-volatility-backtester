"""Build evaluation splits from YAML configuration."""

from __future__ import annotations

from datetime import date

from .splits import DateSplit, TrainValidationTestSplit


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def build_fixed_split(config: dict) -> TrainValidationTestSplit:
    split_config = config["evaluation"]["fixed_split"]
    split = TrainValidationTestSplit(
        train=DateSplit(
            name="train",
            start_date=parse_iso_date(
                split_config["train"]["start"]
            ),
            end_date=parse_iso_date(split_config["train"]["end"]),
            purpose="parameter_estimation",
        ),
        validation=DateSplit(
            name="validation",
            start_date=parse_iso_date(
                split_config["validation"]["start"]
            ),
            end_date=parse_iso_date(
                split_config["validation"]["end"]
            ),
            purpose="model_selection",
        ),
        test=DateSplit(
            name="test",
            start_date=parse_iso_date(split_config["test"]["start"]),
            end_date=parse_iso_date(split_config["test"]["end"]),
            purpose="final_evaluation",
        ),
    )
    split.validate()
    return split
