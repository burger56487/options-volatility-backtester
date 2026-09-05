"""Tests for ATM term-structure and forward-volatility analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.implied_vol.term_structure import (
    _classify_shape,
    compute_term_structure,
)


def make_metrics(ttms, atm_vols):
    return pd.DataFrame(
        {
            "expiry": [
                pd.Timestamp("2026-01-01")
                + pd.Timedelta(days=int(t * 365))
                for t in ttms
            ],
            "time_to_expiry": ttms,
            "atm_vol": atm_vols,
        }
    )


def test_contango():
    result = compute_term_structure(
        make_metrics([0.1, 0.25, 0.5, 1.0], [0.15, 0.18, 0.20, 0.22])
    )
    assert result.shape == "contango"
    assert result.slope > 0
    assert result.annualized_slope > 0
    assert result.relative_slope > 0


def test_backwardation_is_not_calendar_arbitrage():
    result = compute_term_structure(
        make_metrics([0.1, 0.25, 0.5, 1.0], [0.35, 0.28, 0.22, 0.20])
    )
    assert result.shape == "backwardation"
    assert result.calendar_violations == 0


def test_forward_vol_above_spot_when_term_structure_rises():
    result = compute_term_structure(
        make_metrics([0.25, 0.5], [0.20, 0.22])
    )
    row = result.curve.iloc[1]
    assert row["fwd_status"] == "ok"
    assert row["forward_vol"] > 0.22


def test_calendar_violation_detected_with_details():
    result = compute_term_structure(
        make_metrics([0.5, 1.0], [0.30, 0.18])
    )
    assert result.calendar_violations == 1
    assert len(result.violations) == 1
    assert result.violations[0]["fwd_var"] < 0
    assert result.curve.iloc[1]["fwd_status"] == "calendar_violation"
    assert np.isnan(result.curve.iloc[1]["forward_vol"])


def test_tiny_noise_negative_forward_var_is_zero_not_violation():
    # w1 = 0.0200000 (T=0.5, atm=0.2); w2 lower by 2.5e-7, so the
    # half-year forward variance is -5e-7 (inside the 1e-6 tolerance).
    w2 = 0.02 - 2.5e-7
    atm2 = np.sqrt(w2)
    result = compute_term_structure(
        make_metrics([0.5, 1.0], [0.2, atm2])
    )
    assert result.calendar_violations == 0
    assert result.noise_negative_count == 1
    row = result.curve.iloc[1]
    assert row["fwd_status"] == "noise_negative"
    assert row["forward_vol"] == 0.0


def test_insufficient_data():
    result = compute_term_structure(make_metrics([0.25], [0.20]))
    assert result.shape == "insufficient_data"
    assert result.num_valid_expiries == 1


def test_nan_and_duplicate_ttm_handling():
    metrics = make_metrics(
        [0.1, 0.25, 0.25, 0.5, 0.7],
        [0.15, 0.20, 0.21, 0.22, np.nan],
    )
    result = compute_term_structure(metrics)
    assert result.num_valid_expiries == 3
    assert any("去重" in warning for warning in result.warnings)


def test_classify_humped_and_flat():
    assert _classify_shape(np.array([0.15, 0.25, 0.30, 0.22, 0.18])) == (
        "humped"
    )
    assert _classify_shape(np.array([0.20, 0.201, 0.199, 0.20])) == "flat"
