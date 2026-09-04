"""Tests for per-expiry skew metrics and OTM curve construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.implied_vol.skew import (
    MIN_POINTS,
    _safe_interp,
    build_skew_curve,
    compute_delta,
    compute_skew_metrics,
)


def test_safe_interp_basic():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    assert abs(_safe_interp(x, y, 2.5) - 25.0) < 1e-9


def test_safe_interp_unsorted_and_nan():
    x = np.array([3.0, 1.0, np.nan, 2.0])
    y = np.array([30.0, 10.0, 99.0, 20.0])
    assert abs(_safe_interp(x, y, 2.5) - 25.0) < 1e-9


def test_safe_interp_out_of_range_returns_nan():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    assert np.isnan(_safe_interp(x, y, 5.0))


def test_delta_ranges_and_nan_sigma():
    call = compute_delta(100.0, 100.0, 0.25, 0.2, 1.0, True)
    put = compute_delta(100.0, 100.0, 0.25, 0.2, 1.0, False)
    assert 0.4 < call < 0.6
    assert -0.6 < put < -0.4
    assert np.isnan(
        compute_delta(100.0, 100.0, 0.25, np.nan, 1.0, True)
    )
    otm_call = compute_delta(100.0, 120.0, 0.25, 0.2, 1.0, True)
    assert otm_call < call


def _smile_frame(forward: float = 100.0, t: float = 0.25):
    strikes = np.arange(80.0, 121.0, 5.0)
    rows = []
    for strike in strikes:
        lm = np.log(strike / forward)
        iv = 0.20 - 0.15 * lm + 0.3 * lm**2
        preferred = "call" if strike >= forward else "put"
        for option_type in ("call", "put"):
            rows.append(
                {
                    "expiry": pd.Timestamp("2026-01-01"),
                    "strike": strike,
                    "option_type": option_type,
                    "iv_mid": iv,
                    "iv_source_type": preferred,
                    "time_to_expiry": t,
                }
            )
    return pd.DataFrame(rows)


def _forwards(forward: float = 100.0):
    return pd.DataFrame(
        [
            {
                "expiry": pd.Timestamp("2026-01-01"),
                "forward": forward,
                "discount_factor": 0.99,
                "valid": True,
            }
        ]
    )


def test_curve_deduplicates_by_otm_source():
    frame = _smile_frame()
    forward, discount = 100.0, 0.99
    curve = build_skew_curve(frame, forward, 0.25, discount)
    assert len(curve) == 9
    for _, row in curve.iterrows():
        expected = "call" if row["strike"] >= forward else "put"
        assert row["option_type"] == expected
        assert row["option_type"] == row["iv_source_type"]


def test_put_skew_metrics():
    metrics, curves = compute_skew_metrics(_smile_frame(), _forwards())
    assert len(curves) == 1
    row = metrics.iloc[0]
    assert bool(row["valid"])
    assert bool(row["rr_bf_valid"])
    assert row["rr_25"] < 0  # puts richer than calls
    assert row["bf_25"] > 0  # curvature lifts both wings
    assert row["num_points"] == 9
    assert not np.isnan(row["atm_vol"])


def test_single_side_metrics_report_nan_rr():
    frame = _smile_frame()
    frame = frame[frame["strike"] >= 100.0]  # calls only
    metrics, _ = compute_skew_metrics(frame, _forwards())
    row = metrics.iloc[0]
    assert bool(row["valid"])
    assert bool(row["rr_bf_valid"]) is False
    assert np.isnan(row["rr_25"])
    assert np.isnan(row["bf_25"])


def test_insufficient_points_marked_invalid():
    frame = _smile_frame()
    frame = frame[frame["strike"].isin([80.0, 90.0, 100.0])]
    metrics, curves = compute_skew_metrics(frame, _forwards())
    row = metrics.iloc[0]
    assert bool(row["valid"]) is False
    assert len(curves[list(curves)[0]]) < MIN_POINTS


def test_invalid_iv_rows_are_filtered():
    frame = _smile_frame()
    frame.loc[0, "iv_mid"] = np.nan
    frame.loc[1, "iv_mid"] = -0.5
    metrics, curves = compute_skew_metrics(frame, _forwards())
    curve = curves[list(curves)[0]]
    assert (curve["iv_mid"] > 0).all()
    assert metrics.iloc[0]["num_points"] < 9


def test_missing_forward_returns_empty():
    frame = _smile_frame()
    empty = _forwards().query("valid == False")
    metrics, curves = compute_skew_metrics(frame, empty)
    assert metrics.empty
    assert not curves
