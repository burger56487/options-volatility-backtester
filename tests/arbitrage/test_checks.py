"""Unit tests for the five-class chain no-arbitrage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.arbitrage.checks import (
    check_bounds,
    check_butterfly,
    check_calendar,
    check_monotonicity,
    check_parity,
    run_all_checks,
)


def _chain(
    strikes,
    prices,
    *,
    option_type="call",
    expiry="2026-01-01",
    time_to_expiry=0.25,
    spot=100.0,
    quality="good",
):
    frame = pd.DataFrame(
        {
            "expiry": pd.Timestamp(expiry),
            "strike": strikes,
            "option_type": option_type,
            "mid": prices,
            "time_to_expiry": time_to_expiry,
            "spot": spot,
            "quality": quality,
        }
    )
    frame["bid"] = frame["mid"] - 0.02
    frame["ask"] = frame["mid"] + 0.02
    return frame


def _forwards(
    forward: float = 100.0,
    discount: float = 0.98,
    expiry="2026-01-01",
    valid: bool = True,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "expiry": expiry,
                "time_to_expiry": 0.25,
                "forward": forward,
                "discount_factor": discount,
                "valid": valid,
            }
        ]
    )


def test_convex_prices_pass_butterfly():
    frame = _chain([90.0, 100.0, 110.0], [15.0, 8.0, 3.0])
    violations, _ = check_butterfly(frame)
    assert violations == 0


def test_concave_prices_fail_butterfly():
    # Mid hump is the concave (violating) shape.
    frame = _chain([90.0, 100.0, 110.0], [3.0, 12.0, 3.0])
    violations, _ = check_butterfly(frame)
    assert violations > 0


def test_uneven_strikes_convex_prices_pass():
    frame = _chain([90.0, 100.0, 130.0], [15.0, 8.0, 2.0])
    violations, _ = check_butterfly(frame)
    assert violations == 0


def test_call_monotonicity_violation_and_pass():
    bad = _chain([90.0, 100.0, 110.0], [8.0, 10.0, 12.0])
    violations, _ = check_monotonicity(bad)
    assert violations > 0
    good = _chain([90.0, 100.0, 110.0], [15.0, 8.0, 3.0])
    violations, _ = check_monotonicity(good)
    assert violations == 0


def test_calendar_violation():
    frame = pd.DataFrame(
        {
            "expiry": [
                pd.Timestamp("2026-01-01"),
                pd.Timestamp("2026-03-01"),
            ],
            "strike": [100.0, 100.0],
            "option_type": ["call", "call"],
            "mid": [10.0, 8.0],
            "time_to_expiry": [0.1, 0.3],
            "quality": ["good", "good"],
        }
    )
    violations, _ = check_calendar(frame)
    assert violations > 0


def _parity_frame(
    forward=100.0,
    discount=0.98,
    strikes=None,
    corrupt_strike=None,
):
    if strikes is None:
        strikes = np.arange(90.0, 111.0, 5.0)
    rows = []
    for strike in strikes:
        target = discount * (forward - strike)
        put_mid = 20.0
        call_mid = put_mid + target
        if corrupt_strike is not None and strike == corrupt_strike:
            call_mid += 5.0
        rows.extend(
            [
                {
                    "expiry": pd.Timestamp("2026-01-01"),
                    "strike": strike,
                    "option_type": "call",
                    "mid": call_mid,
                    "time_to_expiry": 0.25,
                    "quality": "good",
                },
                {
                    "expiry": pd.Timestamp("2026-01-01"),
                    "strike": strike,
                    "option_type": "put",
                    "mid": put_mid,
                    "time_to_expiry": 0.25,
                    "quality": "good",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_bounds_and_parity_pass_on_clean_chain():
    strikes = np.arange(80.0, 121.0, 5.0)
    rows = []
    for strike in strikes:
        target = 0.98 * (100.0 - strike)
        put_mid = 20.0
        call_mid = put_mid + target
        rows.append(
            {
                "expiry": pd.Timestamp("2026-01-01"),
                "strike": strike,
                "option_type": "call",
                "mid": max(call_mid, 0.0),
                "time_to_expiry": 0.25,
                "quality": "good",
            }
        )
        rows.append(
            {
                "expiry": pd.Timestamp("2026-01-01"),
                "strike": strike,
                "option_type": "put",
                "mid": put_mid,
                "time_to_expiry": 0.25,
                "quality": "good",
            }
        )
    frame = pd.DataFrame(rows)
    forwards = _forwards(expiry="2026-01-01")
    bounds = check_bounds(frame, forwards)
    assert bounds["bound_ok"].all()
    parity_viol, parity = check_parity(frame, forwards)
    assert parity_viol == 0
    assert parity["parity_ok"].all()


def test_parity_catches_corrupt_strike():
    frame = _parity_frame(corrupt_strike=105.0)
    violations, _ = check_parity(frame, _forwards(expiry="2026-01-01"))
    assert violations > 0


def test_missing_valid_forward_is_skipped_not_counted():
    frame = _chain([90.0, 100.0, 110.0], [15.0, 8.0, 3.0])
    forwards = _forwards(expiry="2026-01-01", valid=False)
    bounds = check_bounds(frame, forwards)
    assert bounds["bound_ok"].isna().all()
    report = run_all_checks(frame, forwards)
    assert report.bound_violations == 0


def test_expiry_dtype_mismatch_is_handled():
    frame = _chain([90.0, 100.0], [15.0, 8.0])
    forwards = _forwards(expiry=pd.Timestamp("2026-01-01"))
    bounds = check_bounds(frame, forwards)
    assert bounds["bound_ok"].notna().all()


def test_quality_filter_is_respected():
    frame = _chain([90.0, 100.0, 110.0], [15.0, 8.0, 3.0])
    frame.loc[frame["strike"] == 90.0, "quality"] = "wide_spread"
    bounds = check_bounds(frame, _forwards(expiry="2026-01-01"))
    assert 90.0 not in set(bounds["strike"])
