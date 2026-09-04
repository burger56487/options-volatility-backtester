import numpy as np
import pytest

from src.pricing.black_scholes import option_price
from src.pricing.local_vol import (
    dupire_local_variance,
    local_vol_surface,
)
from src.volatility_surface.surface import SurfacePoint, VolSurface


def _bs_price_matrix(spot, sigma, r, q, t_grid, k_grid):
    return np.array(
        [
            [
                option_price(
                    spot=spot,
                    strike=spot * np.exp(k),
                    time_to_expiry=t,
                    risk_free_rate=r,
                    volatility=sigma,
                    option_type="call",
                    dividend_yield=q,
                )
                for k in k_grid
            ]
            for t in t_grid
        ]
    )


def test_dupire_recovers_constant_volatility():
    t_grid = np.linspace(0.25, 1.0, 9)
    k_grid = np.linspace(-0.08, 0.08, 33)
    prices = _bs_price_matrix(
        spot=100.0,
        sigma=0.2,
        r=0.03,
        q=0.0,
        t_grid=t_grid,
        k_grid=k_grid,
    )
    variance = dupire_local_variance(
        t_grid,
        k_grid,
        prices,
        risk_free_rate=0.03,
        spot=100.0,
    )
    interior = variance[1:-1, 1:-1]
    median = np.nanmedian(interior)
    assert 0.15 < np.sqrt(median) < 0.26


def test_local_vol_surface_smoke(tmp_path):
    from datetime import date, timedelta

    as_of = date(2026, 9, 4)
    surface = VolSurface(
        as_of=as_of,
        source="test",
        points=[
            SurfacePoint(
                expiry=as_of + timedelta(days=d),
                time_to_expiry=d / 365,
                parameters={
                    "a": 0.04,
                    "b": 0.1,
                    "rho": -0.2,
                    "m": 0.0,
                    "sigma": 0.1,
                },
            )
            for d in (30, 60, 90)
        ],
    )
    expiries, moneyness, variance, prices = local_vol_surface(
        surface,
        spot=100.0,
        risk_free_rate=0.04,
        n_expiries=3,
        n_moneyness=11,
    )
    assert variance.shape == (3, 11)
    assert prices.shape == variance.shape
