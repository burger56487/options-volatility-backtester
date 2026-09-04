from datetime import date, timedelta

import pandas as pd

from src.market_data.real_option_chain import (
    add_iv_band,
    estimate_forward_bounds,
    estimate_per_expiry_dividend_yields,
    grade_arbitrage,
)
from src.pricing.black_scholes import option_price


def _quotes():
    snapshot = date(2026, 9, 4)
    rows = []
    for expiry_days in (21, 45):
        expiry = snapshot + timedelta(days=expiry_days)
        for option_type in ("call", "put"):
            for strike in (540.0, 560.0, 580.0):
                mid = option_price(
                    spot=560.0,
                    strike=strike,
                    time_to_expiry=expiry_days / 365.0,
                    risk_free_rate=0.04,
                    volatility=0.25,
                    option_type=option_type,
                    dividend_yield=0.012,
                )
                rows.append(
                    {
                        "expiry": expiry,
                        "time_to_expiry": expiry_days / 365.0,
                        "option_type": option_type,
                        "strike": strike,
                        "bid": mid - 0.05,
                        "ask": mid + 0.05,
                        "spot": 560.0,
                    }
                )
    return pd.DataFrame(rows)


def test_forward_bounds_bracket_single_point_estimate():
    quotes = _quotes()
    bounds = estimate_forward_bounds(quotes, risk_free_rate=0.04)
    assert len(bounds) == 2
    estimates, _ = estimate_per_expiry_dividend_yields(
        quotes,
        risk_free_rate=0.04,
    )
    for expiry, band in bounds.items():
        assert band["forward_low"] <= band["forward_high"]


def test_grading_flags_valid_quotes_usable():
    quotes = _quotes()
    graded = grade_arbitrage(quotes, risk_free_rate=0.04)
    assert graded["usable_for_european_iv"].all()
    assert (~graded["hard_violation"]).all()


def test_iv_band_contains_mid_iv():
    quotes = _quotes()
    banded = add_iv_band(
        quotes,
        risk_free_rate=0.04,
        dividend_yield=0.012,
    )
    clean = banded.dropna(subset=["iv_mid", "iv_bid", "iv_ask"])
    assert (clean["iv_mid"] - 0.25).abs().max() < 1e-4
    assert (clean["iv_bid"] <= clean["iv_mid"]).all()
    assert (clean["iv_mid"] <= clean["iv_ask"]).all()
    assert (clean["iv_band_width"] > 0).all()
