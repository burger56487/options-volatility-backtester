"""Tests for chain Greeks, aggregation and heatmaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.greeks.chain_greeks import (
    aggregate_portfolio_greeks,
    build_greek_heatmap,
    compute_chain_greeks,
    compute_greeks_row,
    top_risk_contracts,
)
from src.pricing.black_scholes import option_price


def test_atm_call_and_put_delta():
    call = compute_greeks_row(100, 100, 1.0, 0.2, 0.02, 0.0, True)
    put = compute_greeks_row(100, 100, 1.0, 0.2, 0.02, 0.0, False)
    assert 0.5 < call["delta"] < 0.65
    assert -0.65 < put["delta"] < -0.35


def test_call_put_delta_relationship():
    q, t = 0.03, 1.0
    call = compute_greeks_row(100, 100, t, 0.2, 0.02, q, True)
    put = compute_greeks_row(100, 100, t, 0.2, 0.02, q, False)
    assert abs((call["delta"] - put["delta"]) - np.exp(-q * t)) < 1e-9


def test_gamma_vega_same_for_call_and_put():
    call = compute_greeks_row(100, 105, 0.5, 0.25, 0.02, 0.0, True)
    put = compute_greeks_row(100, 105, 0.5, 0.25, 0.02, 0.0, False)
    assert abs(call["gamma"] - put["gamma"]) < 1e-12
    assert abs(call["vega"] - put["vega"]) < 1e-12


def test_gamma_vega_positive_and_deep_otm_smaller_gamma():
    atm = compute_greeks_row(100, 100, 0.5, 0.2, 0.02, 0.0, True)
    otm = compute_greeks_row(100, 150, 0.5, 0.2, 0.02, 0.0, True)
    assert atm["gamma"] > 0 and atm["vega"] > 0
    assert otm["gamma"] < atm["gamma"]


def test_invalid_inputs_return_nan():
    assert np.isnan(
        compute_greeks_row(100, 100, 0.0, 0.2, 0.02, 0.0, True)["delta"]
    )
    assert np.isnan(
        compute_greeks_row(
            100, 100, 1.0, np.nan, 0.02, 0.0, True
        )["gamma"]
    )
    assert np.isnan(
        compute_greeks_row(100, 100, 1.0, 0.0, 0.02, 0.0, True)["vega"]
    )


def test_greeks_match_numerical_differentials():
    spot, strike, t, sigma, r, q = 100.0, 100.0, 1.0, 0.2, 0.02, 0.0
    h = 0.01

    def price(s):
        return option_price(
            spot=s,
            strike=strike,
            time_to_expiry=t,
            risk_free_rate=r,
            volatility=sigma,
            option_type="call",
            dividend_yield=q,
        )

    analytical = compute_greeks_row(
        spot, strike, t, sigma, r, q, True
    )
    delta_num = (price(spot + h) - price(spot - h)) / (2 * h)
    gamma_num = (
        price(spot + h) - 2 * price(spot) + price(spot - h)
    ) / h**2
    assert abs(delta_num - analytical["delta"]) < 1e-4
    assert abs(gamma_num - analytical["gamma"]) < 1e-3


def _chain():
    rows = []
    for strike in (90.0, 100.0, 110.0):
        for option_type in ("call", "put"):
            rows.append(
                {
                    "expiry": pd.Timestamp("2026-01-01"),
                    "strike": strike,
                    "option_type": option_type,
                    "spot": 100.0,
                    "time_to_expiry": 0.25,
                    "iv_mid": 0.20,
                    "quality": "good",
                }
            )
    return pd.DataFrame(rows)


def test_chain_computes_all_greeks_and_conversions():
    chain = compute_chain_greeks(_chain())
    assert not chain["delta"].isna().any()
    assert not chain["gamma"].isna().any()
    assert np.allclose(chain["vega_per_pct"], chain["vega"] / 100.0)
    assert np.allclose(chain["theta_per_day"], chain["theta"] / 365.0)


def test_heatmap_gamma_pools_and_delta_requires_type():
    chain = compute_chain_greeks(_chain())
    gamma_map = build_greek_heatmap(chain, "gamma")
    assert gamma_map.shape == (3, 1)
    with pytest.raises(ValueError, match="option_type"):
        build_greek_heatmap(chain, "delta")
    delta_calls = build_greek_heatmap(chain, "delta", option_type="call")
    assert len(delta_calls) == 3


def test_aggregation_and_top_risk():
    chain = compute_chain_greeks(_chain())
    chain["position"] = [2.0, -1.0, 0.0, 3.0, 1.0, 0.0]
    aggregated = aggregate_portfolio_greeks(
        chain,
        position_column="position",
        multiplier=100.0,
    )
    expected_delta = float(
        (chain["delta"] * chain["position"] * 100.0).sum()
    )
    assert abs(aggregated["delta"] - expected_delta) < 1e-9
    assert aggregated["vega_per_pct"] == pytest.approx(
        aggregated["vega"] / 100.0
    )
    top = top_risk_contracts(chain, "gamma", top_n=3)
    assert len(top) == 3
    assert top["gamma"].abs().is_monotonic_decreasing
