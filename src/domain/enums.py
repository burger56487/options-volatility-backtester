"""Single source of truth for shared option/asset enums."""

from __future__ import annotations

from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class DataType(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"
