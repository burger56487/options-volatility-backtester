"""Tests for robust SVI calibration, g(k) and calendar screens."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.volatility_surface.svi import svi_total_variance
from src.volatility_surface.svi_analysis import (
    SVIResult,
    calibrate_svi_curve,
    check_calendar_arbitrage,
    svi_derivatives,
    svi_gatheral_g,
)


TRUE_PARAMS = np.array([0.04, 0.1, -0.3, 0.0, 0.1])


def _curve(k, params=TRUE_PARAMS, time_to_expiry=0.25, nan_at=None):
    w = svi_total_variance(k, *params)
    iv = np.sqrt(w / time_to_expiry)
    if nan_at is not None:
        iv = iv.copy()
        iv[nan_at] = np.nan
    return pd.DataFrame(
        {
            "log_moneyness": k,
            "iv_mid": iv,
        }
    )


def test_svi_recovers_fitted_curve():
    k = np.linspace(-0.3, 0.3, 20)
    result = calibrate_svi_curve(_curve(k), "2026-01-01", 0.25)
    assert result.converged
    w_fit = svi_total_variance(k, *result.params)
    w_true = svi_total_variance(k, *TRUE_PARAMS)
    assert np.max(np.abs(w_fit - w_true)) < 1e-4
    assert result.rmse_vol < 1e-3


def test_derivatives_match_numerical():
    k0 = 0.1
    h = 1e-6
    w, wp, wpp = svi_derivatives(k0, TRUE_PARAMS)
    wp_num = (
        svi_total_variance(k0 + h, *TRUE_PARAMS)
        - svi_total_variance(k0 - h, *TRUE_PARAMS)
    ) / (2 * h)
    wpp_num = (
        svi_total_variance(k0 + h, *TRUE_PARAMS)
        - 2 * svi_total_variance(k0, *TRUE_PARAMS)
        + svi_total_variance(k0 - h, *TRUE_PARAMS)
    ) / h**2
    assert abs(wp - wp_num) < 1e-4
    assert abs(wpp - wpp_num) < 1e-2


def test_valid_svi_has_no_butterfly_beyond_tolerance():
    k = np.linspace(-0.5, 0.5, 200)
    g = svi_gatheral_g(k, TRUE_PARAMS)
    assert np.all(g >= -1e-6)


def test_insufficient_points_invalid():
    curve = pd.DataFrame(
        {
            "log_moneyness": [0.0, 0.1, 0.2],
            "iv_mid": [0.2, 0.19, 0.21],
        }
    )
    result = calibrate_svi_curve(curve, "2026-01-01", 0.25)
    assert not result.valid
    assert not result.converged


def test_nan_filtered():
    k = np.linspace(-0.3, 0.3, 20)
    result = calibrate_svi_curve(
        _curve(k, nan_at=5),
        "2026-01-01",
        0.25,
    )
    assert result.num_points == 19


def _result(expiry, t, a, valid=True):
    return SVIResult(
        expiry=expiry,
        time_to_expiry=t,
        params=np.array([a, 0.1, -0.3, 0.0, 0.1]),
        rmse_vol=1e-4,
        rmse_total_var=1e-5,
        butterfly_violations=0,
        min_g=0.1,
        min_w=0.01,
        converged=True,
        num_points=20,
        valid=valid,
    )


def test_calendar_no_violation():
    violations, _ = check_calendar_arbitrage(
        [_result("e1", 0.25, 0.02), _result("e2", 0.50, 0.05)]
    )
    assert violations == 0


def test_calendar_violation_detected():
    violations, details = check_calendar_arbitrage(
        [_result("e1", 0.25, 0.05), _result("e2", 0.50, 0.02)]
    )
    assert violations > 0
    assert len(details) == 1


def test_invalid_results_excluded_from_calendar_check():
    violations, _ = check_calendar_arbitrage(
        [
            _result("e1", 0.25, 0.05, valid=False),
            _result("e2", 0.50, 0.02, valid=True),
        ]
    )
    assert violations == 0
