"""Dupire local volatility from a vanilla call price grid."""

from __future__ import annotations

import numpy as np

from src.pricing.black_scholes import option_price
from src.volatility_surface.surface import VolSurface


def dupire_local_variance(
    expiry_grid: np.ndarray,
    log_moneyness_grid: np.ndarray,
    call_price_matrix: np.ndarray,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    spot: float = 1.0,
) -> np.ndarray:
    """Local variance from Dupire's formula on a smoothed price grid.

    sigma_loc^2 = 2*(C_T + (r-q) K C_K + q C) / (K^2 C_KK).
    """
    k_grid = np.asarray(log_moneyness_grid, dtype=float)
    t_grid = np.asarray(expiry_grid, dtype=float)
    prices = np.asarray(call_price_matrix, dtype=float)

    c_t = np.gradient(prices, t_grid, axis=0, edge_order=1)
    c_k = np.gradient(prices, k_grid, axis=1, edge_order=1)
    c_kk = np.gradient(c_k, k_grid, axis=1, edge_order=1)

    # Coordinate transform: K = spot*exp(k). Then
    #   K * C_K = C_k
    #   K^2 * C_KK = C_kk - C_k
    second_term = c_kk - c_k
    numerator = 2.0 * (
        c_t
        + (risk_free_rate - dividend_yield) * c_k
        + dividend_yield * prices
    )
    local_variance = np.full_like(prices, np.nan)
    valid = np.isfinite(second_term) & (second_term > 0)
    local_variance[valid] = numerator[valid] / second_term[valid]
    local_variance[local_variance < 0] = np.nan
    return local_variance


def local_vol_surface(
    surface: VolSurface,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    n_expiries: int = 5,
    n_moneyness: int = 21,
    moneyness_width: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a local-variance grid from a VolSurface via SVI-smoothed prices."""
    expiries = np.linspace(
        min(p.time_to_expiry for p in surface.points),
        max(p.time_to_expiry for p in surface.points),
        n_expiries,
    )
    moneyness = np.linspace(
        -moneyness_width,
        moneyness_width,
        n_moneyness,
    )
    prices = np.empty((n_expiries, n_moneyness))
    for i, t in enumerate(expiries):
        for j, k in enumerate(moneyness):
            iv = surface.interpolate_iv(float(k), float(t))
            prices[i, j] = option_price(
                spot=spot,
                strike=spot * np.exp(k),
                time_to_expiry=float(t),
                risk_free_rate=risk_free_rate,
                volatility=iv,
                option_type="call",
                dividend_yield=dividend_yield,
            )
    variance = dupire_local_variance(
        expiries,
        moneyness,
        prices,
        risk_free_rate,
        dividend_yield,
        spot=spot,
    )
    return expiries, moneyness, variance, prices
