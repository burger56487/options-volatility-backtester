"""Tests for the stage-9 liquidity analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.liquidity.analysis import (
    _reliability_score,
    assess_liquidity_state,
    compute_iv_uncertainty,
    compute_spread_metrics,
    liquidity_by_expiry,
    liquidity_by_moneyness,
    rate_reliability,
)


def _make_df(rows):
    return pd.DataFrame(rows)


def test_spread_metrics_tiny_mid_is_nan():
    frame = _make_df(
        [
            {"bid": 1e-6, "ask": 2e-6, "mid": 1.5e-6},
            {"bid": 0.99, "ask": 1.01, "mid": 1.0},
        ]
    )
    work = compute_spread_metrics(frame)
    assert np.isnan(work["rel_spread"].iloc[0])
    assert abs(work["rel_spread"].iloc[1] - 0.02) < 1e-12


def test_vectorized_matches_rowwise():
    frame = _make_df(
        [
            {
                "bid": 0.99,
                "ask": 1.01,
                "mid": 1.0,
                "open_interest": 1000,
                "volume": 500,
                "iv_bid": 0.199,
                "iv_ask": 0.201,
            },
            {
                "bid": 0.5,
                "ask": 1.5,
                "mid": 1.0,
                "open_interest": 5,
                "volume": 0,
                "iv_bid": 0.15,
                "iv_ask": 0.30,
            },
        ]
    )
    frame = compute_spread_metrics(frame)
    frame = compute_iv_uncertainty(frame)
    vectorised = rate_reliability(frame)
    for index in range(len(frame)):
        row_score = _reliability_score(frame.iloc[index])
        assert abs(
            vectorised["reliability_score"].iloc[index] - row_score
        ) < 1e-9


def test_reliability_high_low_medium_bins():
    frame = _make_df(
        [
            {
                "bid": 0.99,
                "ask": 1.01,
                "mid": 1.0,
                "open_interest": 500,
                "volume": 200,
                "iv_bid": 0.199,
                "iv_ask": 0.201,
            },
            {
                "bid": 0.4,
                "ask": 1.6,
                "mid": 1.0,
                "open_interest": 2,
                "volume": 0,
                "iv_bid": 0.10,
                "iv_ask": 0.30,
            },
        ]
    )
    frame = compute_spread_metrics(frame)
    frame = compute_iv_uncertainty(frame)
    rated = rate_reliability(frame)
    assert rated["reliability"].iloc[0] == "high"
    assert rated["reliability"].iloc[1] == "low"


def test_iv_uncertainty_solved_when_band_missing():
    frame = _make_df(
        [
            {
                "bid": 9.5,
                "ask": 10.5,
                "mid": 10.0,
                "strike": 100.0,
                "spot": 100.0,
                "time_to_expiry": 0.25,
                "option_type": "call",
            }
        ]
    )
    work = compute_iv_uncertainty(frame)
    assert work["iv_spread"].notna().all()
    assert (work["iv_spread"] > 0).all()


def test_moneyness_fallback_and_expiry_groups():
    frame = _make_df(
        [
            {
                "expiry": pd.Timestamp("2026-01-01"),
                "strike": 90.0,
                "spot": 100.0,
                "bid": 9.9,
                "ask": 10.1,
                "mid": 10.0,
                "open_interest": 500,
                "volume": 100,
            },
            {
                "expiry": pd.Timestamp("2026-01-01"),
                "strike": 110.0,
                "spot": 100.0,
                "bid": 0.8,
                "ask": 1.2,
                "mid": 1.0,
                "open_interest": 50,
                "volume": 5,
            },
        ]
    )
    by_moneyness = liquidity_by_moneyness(frame)
    by_expiry = liquidity_by_expiry(frame)
    assert not by_moneyness.empty
    assert len(by_expiry) == 1
    assert by_expiry["total_oi"].iloc[0] == 550


def test_assess_uses_full_snapshot_and_reports_good_share():
    frame = _make_df(
        [
            {
                "bid": 0.99,
                "ask": 1.01,
                "mid": 1.0,
                "strike": 100.0,
                "spot": 100.0,
                "time_to_expiry": 0.25,
                "option_type": "call",
                "open_interest": 500,
                "volume": 200,
                "quality": "good",
            },
            {
                "bid": 0.01,
                "ask": 0.99,
                "mid": 0.5,
                "strike": 70.0,
                "spot": 100.0,
                "time_to_expiry": 0.25,
                "option_type": "put",
                "open_interest": 0,
                "volume": 0,
                "quality": "out_of_range",
            },
        ]
    )
    state = assess_liquidity_state(frame)
    assert state.total_open_interest == 500
    assert state.pct_good == 50.0
    assert state.overall_state in {
        "liquid",
        "moderate",
        "illiquid",
        "unknown",
    }
