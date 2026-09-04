import math
from datetime import date, timedelta

import pandas as pd

from src.market_data.real_option_chain import (
    add_implied_volatility,
    clean_quote_frame,
)
from src.pricing.black_scholes import option_price


def _quote_row(
    option_type: str,
    strike: float,
    snapshot: date,
    expiry_days: int,
    volatility: float = 0.25,
) -> dict:
    expiry = snapshot + timedelta(days=expiry_days)
    time_to_expiry = expiry_days / 365.0
    mid = option_price(
        spot=560.0,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=0.04,
        volatility=volatility,
        option_type=option_type,
        dividend_yield=0.012,
    )
    return {
        "snapshot_date": pd.Timestamp(snapshot),
        "expiry": pd.Timestamp(expiry),
        "option_type": option_type,
        "strike": strike,
        "bid": mid - 0.05,
        "ask": mid + 0.05,
        "spot": 560.0,
        "volume": 100,
        "open_interest": 1000,
    }


def test_clean_quote_frame_filters_invalid_quotes():
    snapshot = date(2026, 9, 4)
    rows = [
        _quote_row("call", 560.0, snapshot, 30),
        _quote_row("put", 560.0, snapshot, 30),
        _quote_row("call", 540.0, snapshot, 60),
        _quote_row("put", 580.0, snapshot, 60),
    ]
    rows.append(
        {
            **_quote_row("call", 560.0, snapshot, 30),
            "expiry": pd.Timestamp(snapshot),
        }
    )
    rows.append(
        {
            **_quote_row("put", 560.0, snapshot, 30),
            "bid": 3.0,
            "ask": 2.0,
        }
    )

    quotes = pd.DataFrame(rows)
    cleaned, report = clean_quote_frame(quotes)

    assert len(cleaned) == 4
    assert report["dropped_rows"] == 2
    assert "expired" in report["drop_reasons"]
    assert "ask_below_bid" in report["drop_reasons"]


def test_implied_volatility_roundtrip_from_model_mid():
    snapshot = date(2026, 9, 4)
    rows = []
    for expiry_days in (21, 45):
        for option_type in ("call", "put"):
            for strike in (520.0, 540.0, 560.0, 580.0, 600.0):
                rows.append(
                    _quote_row(
                        option_type=option_type,
                        strike=strike,
                        snapshot=snapshot,
                        expiry_days=expiry_days,
                    )
                )

    quotes = pd.DataFrame(rows)
    cleaned, _ = clean_quote_frame(quotes)
    with_iv = add_implied_volatility(
        cleaned,
        risk_free_rate=0.04,
        dividend_yield=0.012,
    )

    solved = with_iv.dropna(subset=["iv"])
    assert len(solved) == len(with_iv)
    assert (solved["iv_error"] == "").all()
    assert (solved["iv"] - 0.25).abs().max() < 1e-4
    assert with_iv["log_moneyness"].notna().all()


def test_dividend_yield_estimated_from_atm_pairs():
    snapshot = date(2026, 9, 4)
    rows = []
    for expiry_days in (21, 45):
        for option_type in ("call", "put"):
            for strike in (540.0, 560.0, 580.0):
                rows.append(
                    _quote_row(
                        option_type=option_type,
                        strike=strike,
                        snapshot=snapshot,
                        expiry_days=expiry_days,
                    )
                )
    quotes = pd.DataFrame(rows)
    cleaned, _ = clean_quote_frame(quotes)
    with_iv = add_implied_volatility(
        cleaned,
        risk_free_rate=0.04,
        dividend_yield=None,
    )
    estimated = with_iv["dividend_yield_used"].unique()
    assert (estimated - 0.012).max() < 0.002
    assert (with_iv["iv"] - 0.25).abs().max() < 1e-4
