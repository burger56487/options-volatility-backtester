"""Unit tests for five-class quote quality grading."""

from __future__ import annotations

import pandas as pd

from src.market_data.quality_grading import (
    QualityGrade,
    clean_graded_subset,
    generate_grading_report,
    grade_quote_quality,
)


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a frame with every column the grader can consult."""
    return pd.DataFrame(
        [
            {
                "bid": None,
                "ask": None,
                "strike": None,
                "spot": None,
                "volume": None,
                "open_interest": None,
                "option_type": None,
                "expiry": None,
                "snapshot_date": None,
                **row,
            }
            for row in rows
        ]
    )


def test_crossed_quote_flagged_invalid():
    frame = make_frame(
        [
            {
                "bid": 5.0,
                "ask": 4.0,
                "strike": 100.0,
                "spot": 100.0,
                "volume": 10.0,
                "open_interest": 20.0,
            }
        ]
    )
    graded = grade_quote_quality(frame)
    assert graded.loc[0, "quality"] == QualityGrade.INVALID.value


def test_negative_bid_flagged_invalid():
    frame = make_frame(
        [
            {
                "bid": -1.0,
                "ask": 4.0,
                "strike": 100.0,
                "spot": 100.0,
                "volume": 10.0,
                "open_interest": 20.0,
            }
        ]
    )
    graded = grade_quote_quality(frame)
    assert graded.loc[0, "quality"] == QualityGrade.INVALID.value


def test_missing_ask_flagged_invalid():
    frame = make_frame(
        [
            {
                "bid": 1.0,
                "ask": float("nan"),
                "strike": 100.0,
                "spot": 100.0,
            }
        ]
    )
    graded = grade_quote_quality(frame)
    assert graded.loc[0, "quality"] == QualityGrade.INVALID.value


def test_wide_spread_flagged():
    frame = make_frame(
        [
            {
                "bid": 1.0,
                "ask": 3.0,
                "strike": 100.0,
                "spot": 100.0,
                "volume": 1.0,
                "open_interest": 5.0,
            }
        ]
    )
    graded = grade_quote_quality(frame)
    assert graded.loc[0, "quality"] == QualityGrade.WIDE_SPREAD.value


def test_penny_tick_quote_is_not_wide():
    frame = make_frame(
        [
            {
                "bid": 0.01,
                "ask": 0.02,
                "strike": 100.0,
                "spot": 100.0,
                "volume": 1.0,
                "open_interest": 5.0,
            }
        ]
    )
    graded = grade_quote_quality(frame)
    assert graded.loc[0, "quality"] == QualityGrade.GOOD.value


def test_good_quote_passes():
    frame = make_frame(
        [
            {
                "bid": 1.95,
                "ask": 2.05,
                "strike": 100.0,
                "spot": 100.0,
                "volume": 1.0,
                "open_interest": 5.0,
            }
        ]
    )
    graded = grade_quote_quality(frame)
    assert graded.loc[0, "quality"] == QualityGrade.GOOD.value
    assert len(clean_graded_subset(graded)) == 1


def test_low_liquidity_flagged():
    frame = make_frame(
        [
            {
                "bid": 1.95,
                "ask": 2.05,
                "strike": 100.0,
                "spot": 100.0,
                "volume": 0.0,
                "open_interest": 0.0,
            }
        ]
    )
    graded = grade_quote_quality(frame)
    assert graded.loc[0, "quality"] == QualityGrade.LOW_LIQUIDITY.value


def test_out_of_range_flagged():
    frame = make_frame(
        [
            {
                "bid": 1.0,
                "ask": 1.1,
                "strike": 300.0,
                "spot": 100.0,
                "volume": 1.0,
                "open_interest": 5.0,
            }
        ]
    )
    graded = grade_quote_quality(frame)
    assert graded.loc[0, "quality"] == QualityGrade.OUT_OF_RANGE.value


def test_invalid_priority_over_out_of_range():
    frame = make_frame(
        [
            {
                "bid": -1.0,
                "ask": 4.0,
                "strike": 300.0,
                "spot": 100.0,
                "volume": 1.0,
                "open_interest": 5.0,
            }
        ]
    )
    graded = grade_quote_quality(frame)
    assert graded.loc[0, "quality"] == QualityGrade.INVALID.value


def test_out_of_range_priority_over_low_liquidity():
    frame = make_frame(
        [
            {
                "bid": 1.0,
                "ask": 1.1,
                "strike": 300.0,
                "spot": 100.0,
                "volume": 0.0,
                "open_interest": 0.0,
            }
        ]
    )
    graded = grade_quote_quality(frame)
    assert graded.loc[0, "quality"] == QualityGrade.OUT_OF_RANGE.value


def test_grading_report_counts():
    frame = make_frame(
        [
            {
                "bid": 1.95,
                "ask": 2.05,
                "strike": 100.0,
                "spot": 100.0,
                "volume": 1.0,
                "open_interest": 5.0,
                "option_type": "call",
                "expiry": "2026-09-10",
                "snapshot_date": "2026-09-04",
            },
            {
                "bid": 1.0,
                "ask": 3.0,
                "strike": 100.0,
                "spot": 100.0,
                "volume": 1.0,
                "open_interest": 5.0,
                "option_type": "put",
                "expiry": "2026-09-10",
                "snapshot_date": "2026-09-04",
            },
            {
                "bid": -1.0,
                "ask": 4.0,
                "strike": 100.0,
                "spot": 100.0,
                "volume": 1.0,
                "open_interest": 5.0,
                "option_type": "put",
                "expiry": "2026-09-10",
                "snapshot_date": "2026-09-04",
            },
        ]
    )
    graded = grade_quote_quality(frame)
    report = generate_grading_report(graded)
    assert report["total_contracts"] == 3
    assert report["by_quality"] == {
        "good": 1,
        "wide_spread": 1,
        "low_liquidity": 0,
        "out_of_range": 0,
        "invalid": 1,
    }
    assert report["good_pct"] == round(100.0 / 3, 2)
    assert report["num_calls"] == 1
    assert report["num_puts"] == 2
    assert report["num_expiries"] == 1
