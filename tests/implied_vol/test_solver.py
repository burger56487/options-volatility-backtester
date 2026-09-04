"""Tests for Black-76 implied-volatility solving and chain solving."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.implied_vol.solver import (
    MIN_TTM,
    black76_price,
    black76_vega,
    implied_vol,
    solve_chain_iv,
)


def test_iv_roundtrip_atm():
    forward, strike, t, discount, sigma = 100.0, 100.0, 0.25, 0.99, 0.20
    price = black76_price(forward, strike, t, sigma, discount, True)
    iv, status = implied_vol(price, forward, strike, t, discount, True)
    assert status == "ok"
    assert abs(iv - sigma) < 1e-6


def test_iv_roundtrip_otm_call_and_put():
    for strike, is_call in ((120.0, True), (90.0, False)):
        forward, t, discount, sigma = 100.0, 0.5, 0.98, 0.35
        price = black76_price(
            forward, strike, t, sigma, discount, is_call
        )
        iv, status = implied_vol(
            price, forward, strike, t, discount, is_call
        )
        assert status == "ok"
        assert abs(iv - sigma) < 1e-6


def test_iv_roundtrip_high_vol():
    forward, strike, t, discount, sigma = 100.0, 100.0, 1.0, 0.98, 1.5
    price = black76_price(forward, strike, t, sigma, discount, True)
    iv, status = implied_vol(price, forward, strike, t, discount, True)
    assert status == "ok"
    assert abs(iv - sigma) < 1e-5


def test_deep_itm_with_time_value_recovers():
    forward, strike, t, discount, sigma = 100.0, 70.0, 1.0, 0.98, 0.30
    price = black76_price(forward, strike, t, sigma, discount, True)
    iv, status = implied_vol(price, forward, strike, t, discount, True)
    assert status == "ok"
    assert abs(iv - sigma) < 1e-3


def test_price_below_intrinsic_is_no_arb():
    forward, strike, t, discount = 100.0, 90.0, 0.25, 0.99
    iv, status = implied_vol(
        discount * (forward - strike) - 1.0,
        forward,
        strike,
        t,
        discount,
        True,
    )
    assert status == "no_arb"
    assert np.isnan(iv)


def test_price_at_intrinsic_returns_min_vol():
    forward, strike, t, discount = 100.0, 90.0, 0.25, 0.99
    lower = discount * (forward - strike)
    iv, status = implied_vol(
        lower,
        forward,
        strike,
        t,
        discount,
        True,
    )
    assert status == "at_intrinsic"
    assert iv > 0


def test_price_above_upper_is_no_arb():
    forward, strike, t, discount = 100.0, 100.0, 0.25, 0.99
    upper = discount * forward
    iv, status = implied_vol(
        upper + 1.0,
        forward,
        strike,
        t,
        discount,
        True,
    )
    assert status == "no_arb"


def test_upper_bound_epsilon_band_not_mislabeled_no_arb():
    forward, strike, t, discount = 100.0, 99.99, 1.0, 0.99
    upper = discount * forward
    iv, status = implied_vol(
        upper - 5e-9,
        forward,
        strike,
        t,
        discount,
        True,
    )
    assert status != "no_arb"


def test_expired_and_nan_price_handling():
    iv, status = implied_vol(5.0, 100.0, 95.0, 0.0, 1.0, True)
    assert status == "expired"
    assert np.isnan(iv)
    iv, status = implied_vol(
        float("nan"), 100.0, 95.0, MIN_TTM * 2, 1.0, True
    )
    assert status == "no_arb"
    assert np.isnan(iv)


def test_vega_positive():
    assert black76_vega(100.0, 100.0, 0.25, 0.2, 0.99) > 0


def _chain_with_sigma(sigma=0.25):
    forward, t, discount = 100.0, 0.5, 0.98
    rows = []
    for strike in (90.0, 100.0, 110.0):
        for option_type, is_call in (("call", True), ("put", False)):
            mid = black76_price(
                forward,
                strike,
                t,
                sigma,
                discount,
                is_call,
            )
            rows.append(
                {
                    "expiry": pd.Timestamp("2026-01-01"),
                    "strike": strike,
                    "option_type": option_type,
                    "mid": mid,
                    "bid": mid - 0.01,
                    "ask": mid + 0.01,
                    "time_to_expiry": t,
                    "quality": "good",
                }
            )
    return pd.DataFrame(rows)


def _forwards():
    return pd.DataFrame(
        [
            {
                "expiry": pd.Timestamp("2026-01-01"),
                "forward": 100.0,
                "discount_factor": 0.98,
                "valid": True,
            }
        ]
    )


def test_solve_chain_otm_unified_mid_and_own_quotes():
    frame = _chain_with_sigma()
    solved = solve_chain_iv(frame, _forwards())
    assert len(solved) == 6
    for _, row in solved.iterrows():
        assert (solved["iv_status"] == "ok").all()
        assert abs(row["iv_mid"] - 0.25) < 1e-6
        expected_source = "call" if row["strike"] >= 100.0 else "put"
        assert row["iv_source_type"] == expected_source
        # Own-quote vols bracket the OTM-mid vol.
        assert row["iv_bid"] <= row["iv_mid"] + 1e-6
        assert row["iv_ask"] >= row["iv_mid"] - 1e-6


def test_solve_chain_synthesises_missing_otm_side():
    frame = _chain_with_sigma()
    # Keep only the call at K=90 so the put (preferred OTM side) is missing.
    frame = frame[
        ~(
            (frame["strike"] == 90.0)
            & (frame["option_type"] == "put")
        )
    ]
    solved = solve_chain_iv(frame, _forwards())
    row = solved[
        (solved["strike"] == 90.0)
        & (solved["option_type"] == "call")
    ].iloc[0]
    assert row["iv_status"] == "ok"
    assert row["iv_source_type"] == "put"
    assert abs(row["iv_mid"] - 0.25) < 1e-6


def test_solve_chain_no_forward_and_bad_quality():
    frame = _chain_with_sigma()
    empty = _forwards().query("valid == False")
    solved = solve_chain_iv(frame, empty)
    assert (solved["iv_status"] == "no_forward").all()

    frame_no_quality = frame.drop(columns=["quality"])
    with pytest.raises(ValueError, match="quality"):
        solve_chain_iv(frame_no_quality, _forwards())
