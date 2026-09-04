"""Leg-level daily attribution and waterfall PnL attribution."""

from __future__ import annotations

import math

from src.pricing.black_scholes import option_price


def attribute_daily_leg(
    prev: dict,
    spot_change: float,
    iv_change: float,
    dt: float,
    risk_free_change: float = 0.0,
) -> dict[str, float]:
    """Second-order daily attribution for one position.

    ``prev`` provides per-position delta/gamma/vega/theta/rho (all already
    scaled by quantity and multiplier). Contributions approximate the change
    in value from spot, gamma, vega, theta and rho.
    """
    return {
        "delta": prev["delta"] * spot_change,
        "gamma": 0.5 * prev["gamma"] * spot_change**2,
        "vega": prev["vega"] * iv_change,
        "theta": prev["theta"] * dt,
        "rho": prev["rho"] * risk_free_change,
    }


def _bsm(spot, strike, t, r, volatility, option_type, q=0.0):
    return option_price(
        spot=spot,
        strike=strike,
        time_to_expiry=t,
        risk_free_rate=r,
        volatility=volatility,
        option_type=option_type,
        dividend_yield=q,
    )


def waterfall_attribution(
    prev_spot: float,
    new_spot: float,
    prev_iv: float,
    new_iv: float,
    prev_t: float,
    new_t: float,
    prev_r: float,
    new_r: float,
    strike: float,
    option_type: str,
    dividend_yield: float = 0.0,
    order: tuple[str, ...] = ("spot", "volatility", "time", "rate"),
) -> dict:
    """Exact-revaluation waterfall: sequentially replace risk factors."""
    state = {
        "spot": prev_spot,
        "volatility": prev_iv,
        "time": prev_t,
        "rate": prev_r,
    }
    contributions = {}
    previous_value = _bsm(
        state["spot"],
        strike,
        state["time"],
        state["rate"],
        state["volatility"],
        option_type,
        dividend_yield,
    )
    updates = {
        "spot": new_spot,
        "volatility": new_iv,
        "time": new_t,
        "rate": new_r,
    }
    for factor in order:
        state[factor] = updates[factor]
        value = _bsm(
            state["spot"],
            strike,
            state["time"],
            state["rate"],
            state["volatility"],
            option_type,
            dividend_yield,
        )
        contributions[factor] = value - previous_value
        previous_value = value
    final_value = _bsm(
        new_spot,
        strike,
        new_t,
        new_r,
        new_iv,
        option_type,
        dividend_yield,
    )
    return {
        "contributions": contributions,
        "actual_change": final_value
        - _bsm(
            prev_spot,
            strike,
            prev_t,
            prev_r,
            prev_iv,
            option_type,
            dividend_yield,
        ),
        "attributed_change": sum(contributions.values()),
        "order": list(order),
    }


def decompose_iv_change(
    k_grid,
    iv_before,
    iv_after,
) -> dict[str, float]:
    """Split an IV curve move into level, skew and curvature changes."""
    import numpy as np

    k = np.asarray(k_grid, dtype=float)
    before = np.asarray(iv_before, dtype=float)
    after = np.asarray(iv_after, dtype=float)
    if k.ndim != 1 or before.ndim != 1 or after.ndim != 1:
        raise ValueError("k_grid and IV arrays must be one-dimensional.")
    if not (len(k) == len(before) == len(after)):
        raise ValueError("k_grid and IV arrays must have equal length.")
    if len(k) < 3:
        raise ValueError("At least 3 strikes are required for quadratic fit.")
    if not np.all(np.isfinite(k)) or not np.all(
        np.isfinite(before)
    ) or not np.all(np.isfinite(after)):
        raise ValueError("k_grid and IV arrays must contain only finite values.")
    centered_k = k - np.mean(k)
    poly_before = np.polyfit(centered_k, before, 2)
    poly_after = np.polyfit(centered_k, after, 2)
    return {
        "level_change": float(poly_after[2] - poly_before[2]),
        "skew_change": float(poly_after[1] - poly_before[1]),
        "curvature_change": float(poly_after[0] - poly_before[0]),
    }
