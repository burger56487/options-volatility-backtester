"""Tests for regression-based forward/discount/rate estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.implied_vol.forward import (
    build_call_put_pairs,
    estimate_all_forwards,
    estimate_forward_single_expiry,
)


def _raw_frame(
    *,
    forward: float = 100.0,
    discount: float = 0.98,
    time_to_expiry: float = 0.25,
    spot: float = 99.0,
    strikes: np.ndarray | None = None,
    outlier_strike: float | None = None,
    quality: str = "good",
) -> pd.DataFrame:
    if strikes is None:
        strikes = np.arange(80.0, 121.0, 5.0)
    rows = []
    for strike in strikes:
        delta = discount * (forward - strike)
        put_mid = 20.0
        call_mid = put_mid + delta
        if outlier_strike is not None and strike == outlier_strike:
            call_mid += 999.0
        for option_type, mid in (("call", call_mid), ("put", put_mid)):
            rows.append(
                {
                    "expiry": pd.Timestamp("2026-01-01"),
                    "strike": strike,
                    "option_type": option_type,
                    "mid": mid,
                    "bid": mid - 0.01,
                    "ask": mid + 0.01,
                    "time_to_expiry": time_to_expiry,
                    "spot": spot,
                    "quality": quality,
                }
            )
    return pd.DataFrame(rows)


def test_recovers_forward_and_discount_through_pipeline():
    frame = _raw_frame()
    estimates = estimate_all_forwards(frame)
    assert len(estimates) == 1
    row = estimates.iloc[0]
    assert bool(row["valid"]) is True
    assert abs(row["discount_factor"] - 0.98) < 1e-6
    assert abs(row["forward"] - 100.0) < 1e-6
    assert row["r_squared"] > 0.9999
    assert row["num_pairs_used"] == 9


def test_insufficient_pairs_is_invalid():
    frame = _raw_frame(strikes=np.array([100.0, 105.0]))
    pairs = build_call_put_pairs(frame)
    result = estimate_forward_single_expiry(pairs)
    assert result.valid is False
    assert "配对点不足" in " ".join(result.warnings)


def test_outlier_is_removed_and_forward_recovered():
    frame = _raw_frame(outlier_strike=110.0)
    estimates = estimate_all_forwards(frame)
    row = estimates.iloc[0]
    assert bool(row["valid"]) is True
    assert row["outliers_removed"] == 1
    assert abs(row["forward"] - 100.0) < 0.5


def test_wrong_slope_sign_is_invalid_not_crash():
    strikes = np.arange(80.0, 121.0, 5.0)
    frame = _raw_frame(strikes=strikes)
    # Flip the parity sign so C-P increases with K (positive slope).
    calls = frame["option_type"] == "call"
    frame.loc[calls, "mid"] = (
        20.0 + 0.98 * (frame.loc[calls, "strike"] - 100.0)
    )
    pairs = build_call_put_pairs(frame)
    result = estimate_forward_single_expiry(pairs)
    assert result.valid is False
    assert "贴现因子非正" in " ".join(result.warnings)


def test_quality_filter_excludes_wide_quotes():
    frame = _raw_frame()
    # Mark the 105 strike as wide spread.
    mask = frame["strike"] == 105.0
    frame.loc[mask, "quality"] = "wide_spread"
    filtered = build_call_put_pairs(frame)
    assert 105.0 not in set(filtered["strike"])
    unfiltered = build_call_put_pairs(frame, good_only=False)
    assert 105.0 in set(unfiltered["strike"])


def test_absurd_implied_rate_marks_invalid():
    frame = _raw_frame(
        discount=0.5,
        time_to_expiry=0.005,
    )
    estimates = estimate_all_forwards(frame)
    row = estimates.iloc[0]
    assert bool(row["valid"]) is False
    assert "绝对值异常" in " ".join(row["warnings"])


def test_out_of_range_discount_marks_invalid_with_values():
    frame = _raw_frame(discount=1.10)
    estimates = estimate_all_forwards(frame)
    row = estimates.iloc[0]
    assert bool(row["valid"]) is False
    assert "贴现因子超出合理范围" in " ".join(row["warnings"])
