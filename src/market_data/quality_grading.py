"""Per-row quality grading for option-chain snapshots.

Every quote is labelled instead of dropped, so the same frame can be used
for diagnostics and for analysis on the clean subset. Priority order (high
wins): invalid > out_of_range > low_liquidity > wide_spread > good.

The taxonomy keeps the project's published moneyness convention
(|log-moneyness| <= 0.15 for the analysis universe) and the quote-level
structural rules used by :func:`clean_quote_frame`.
"""

from __future__ import annotations

import math
from enum import Enum

import pandas as pd

from .spread_rules import (
    DEFAULT_MAX_RELATIVE_SPREAD,
    DEFAULT_MINIMUM_ABSOLUTE_SPREAD,
    wide_spread_mask,
)


class QualityGrade(str, Enum):
    GOOD = "good"
    WIDE_SPREAD = "wide_spread"
    LOW_LIQUIDITY = "low_liquidity"
    OUT_OF_RANGE = "out_of_range"
    INVALID = "invalid"


# Grades ordered from lowest to highest priority; later assignments override
# earlier ones so a quote always carries its most severe label.
_GRADE_ORDER = [
    QualityGrade.WIDE_SPREAD,
    QualityGrade.LOW_LIQUIDITY,
    QualityGrade.OUT_OF_RANGE,
    QualityGrade.INVALID,
]


def _column(frame: pd.DataFrame, name: str) -> pd.Series | None:
    return frame[name] if name in frame.columns else None


def grade_quote_quality(
    frame: pd.DataFrame,
    *,
    max_relative_spread: float = DEFAULT_MAX_RELATIVE_SPREAD,
    minimum_absolute_spread: float = DEFAULT_MINIMUM_ABSOLUTE_SPREAD,
    max_abs_log_moneyness: float = 0.15,
) -> pd.DataFrame:
    """Return a copy of *frame* with a ``quality`` column added.

    Rules:

    - ``invalid``: missing ask/bid/strike, negative bid, non-positive ask,
      crossed quote, non-positive mid or expired contract.
    - ``out_of_range``: |log-moneyness| above ``max_abs_log_moneyness``.
      Uses the ``log_moneyness`` column when present and otherwise computes
      log(strike / spot).
    - ``low_liquidity``: zero volume and zero open interest (missing volume
      or open interest is treated as zero, matching the normalisation used
      by the real-chain pipeline).
    - ``wide_spread``: positive mid with relative spread above
      ``max_relative_spread``.
    """
    df = frame.copy()
    df["quality"] = QualityGrade.GOOD.value

    mid = _column(df, "mid")
    if mid is None:
        mid = 0.5 * (df["bid"] + df["ask"])
        df["mid"] = mid

    invalid = (
        df["bid"].isna()
        | df["ask"].isna()
        | df["strike"].isna()
        | (df["bid"] < 0)
        | (df["ask"] <= 0)
        | (df["ask"] < df["bid"])
        | (mid <= 0)
        | (df["strike"] <= 0)
    )
    expiry = _column(df, "expiry")
    snapshot_date = _column(df, "snapshot_date")
    if expiry is not None and snapshot_date is not None:
        invalid |= pd.to_datetime(expiry) <= pd.to_datetime(snapshot_date)

    log_moneyness = _column(df, "log_moneyness")
    if log_moneyness is None:
        log_moneyness = (df["strike"] / df["spot"]).apply(math.log)
    out_of_range = log_moneyness.abs() > max_abs_log_moneyness

    volume = _column(df, "volume")
    open_interest = _column(df, "open_interest")
    if volume is None or open_interest is None:
        low_liquidity = pd.Series(False, index=df.index)
    else:
        low_liquidity = (
            volume.fillna(0.0).astype(float) == 0.0
        ) & (open_interest.fillna(0.0).astype(float) == 0.0)

    wide_spread = wide_spread_mask(
        df["ask"] - df["bid"],
        mid,
        max_relative_spread=max_relative_spread,
        minimum_absolute_spread=minimum_absolute_spread,
    )

    for grade in _GRADE_ORDER:
        if grade is QualityGrade.WIDE_SPREAD:
            mask = wide_spread
        elif grade is QualityGrade.LOW_LIQUIDITY:
            mask = low_liquidity
        elif grade is QualityGrade.OUT_OF_RANGE:
            mask = out_of_range
        else:
            mask = invalid
        df.loc[mask, "quality"] = grade.value

    return df


def clean_graded_subset(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only quotes graded ``good`` for downstream analysis."""
    return frame[frame["quality"] == QualityGrade.GOOD.value].copy()


def generate_grading_report(frame: pd.DataFrame) -> dict:
    """Return per-grade counts and contract-level statistics."""
    total = int(len(frame))
    counts = frame["quality"].value_counts()
    by_quality = {
        grade.value: int(counts.get(grade.value, 0))
        for grade in QualityGrade
    }
    report = {
        "total_contracts": total,
        "by_quality": by_quality,
        "good_pct": round(100.0 * by_quality["good"] / total, 2)
        if total
        else 0.0,
    }

    option_type = _column(frame, "option_type")
    if option_type is not None:
        report["num_calls"] = int(
            (option_type.astype(str).str.lower() == "call").sum()
        )
        report["num_puts"] = int(
            (option_type.astype(str).str.lower() == "put").sum()
        )
    expiry = _column(frame, "expiry")
    if expiry is not None:
        report["num_expiries"] = int(pd.to_datetime(expiry).nunique())

    return report
