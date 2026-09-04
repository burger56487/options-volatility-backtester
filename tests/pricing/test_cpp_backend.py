import math

import numpy as np
import pytest

from src.pricing.black_scholes import option_price
from src.pricing.cpp_backend import (
    cpp_batch_bs,
    cpp_mc_gbm,
    cpp_mc_gbm_parallel,
    cpp_portfolio_var,
    cpp_scenario_pnl,
    cpp_surface_prices,
    is_available,
)
from src.risk.contributions import linear_risk_contributions
from src.stochastic.processes import simulate_gbm_terminal


pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="C++ backend DLL not built",
)


def test_cpp_batch_bs_matches_python():
    spot = np.array([100.0, 90.0, 110.0])
    strike = np.array([100.0, 95.0, 105.0])
    t = np.array([0.5, 1.0, 0.25])
    vol = np.array([0.2, 0.3, 0.25])
    option_type = np.array(["call", "put", "call"])
    out = cpp_batch_bs(
        spot, strike, t, 0.04, 0.01, vol, option_type
    )
    for i in range(3):
        reference = option_price(
            spot=float(spot[i]),
            strike=float(strike[i]),
            time_to_expiry=float(t[i]),
            risk_free_rate=0.04,
            volatility=float(vol[i]),
            option_type=str(option_type[i]),
            dividend_yield=0.01,
        )
        assert abs(out["price"][i] - reference) < 1e-10


def test_cpp_mc_overlaps_python_mc_ci():
    params = dict(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        volatility=0.25,
    )
    cpp = cpp_mc_gbm(
        **params,
        n_paths=200_000,
        option_type="call",
        seed=1,
    )
    reference = option_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        volatility=0.25,
        option_type="call",
        dividend_yield=0.01,
    )
    assert abs(cpp["price"] - reference) < 3 * cpp["standard_error"]


def test_cpp_mc_matches_python_path_distribution():
    terminal = simulate_gbm_terminal(
        spot=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        volatility=0.25,
        n_paths=200_000,
        dividend_yield=0.01,
        seed=2,
    )
    discount = math.exp(-0.04 * 0.5)
    py_price = float(np.mean(discount * np.maximum(terminal - 100.0, 0.0)))
    py_se = float(np.std(discount * np.maximum(terminal - 100.0, 0.0), ddof=1) / math.sqrt(len(terminal)))
    cpp = cpp_mc_gbm(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        volatility=0.25,
        n_paths=200_000,
        option_type="call",
        seed=2,
    )
    # Different RNGs: both estimates should be within a few combined SEs.
    combined_se = math.sqrt(py_se**2 + cpp["standard_error"] ** 2)
    assert abs(py_price - cpp["price"]) < 3 * combined_se


def test_cpp_scenario_pnl_matches_python():
    spot = np.array([100.0, 90.0, 110.0])
    strike = np.array([100.0, 95.0, 105.0])
    t = np.array([0.5, 1.0, 0.25])
    vol = np.array([0.2, 0.3, 0.25])
    option_type = np.array(["call", "put", "call"])
    cpp_pnl = cpp_scenario_pnl(
        spot,
        strike,
        t,
        0.04,
        0.01,
        vol,
        option_type,
        spot_shock=-0.1,
        vol_shock=0.05,
    )
    for i in range(3):
        base = option_price(
            spot=float(spot[i]),
            strike=float(strike[i]),
            time_to_expiry=float(t[i]),
            risk_free_rate=0.04,
            volatility=float(vol[i]),
            option_type=str(option_type[i]),
            dividend_yield=0.01,
        )
        shocked = option_price(
            spot=float(spot[i]) * 0.9,
            strike=float(strike[i]),
            time_to_expiry=float(t[i]),
            risk_free_rate=0.04,
            volatility=float(vol[i]) + 0.05,
            option_type=str(option_type[i]),
            dividend_yield=0.01,
        )
        assert abs(cpp_pnl[i] - (shocked - base)) < 1e-10


def test_cpp_portfolio_var_matches_python_euler():
    exposures = np.array([100.0, -50.0])
    covariance = np.array([[0.01, 0.0], [0.0, 0.04]])
    cpp = cpp_portfolio_var(exposures, covariance, z_score=1.645)
    py = linear_risk_contributions(
        exposures, covariance, z_score=1.645
    )
    assert abs(cpp["var"] - py["portfolio_var"]) < 1e-9
    assert np.allclose(
        cpp["contributions"], py["contributions"], atol=1e-9
    )


def test_cpp_parallel_mc_matches_single_thread():
    serial = cpp_mc_gbm(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        volatility=0.25,
        n_paths=200_000,
        option_type="call",
        seed=1,
    )
    parallel = cpp_mc_gbm_parallel(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        volatility=0.25,
        n_paths=200_000,
        option_type="call",
        seed=1,
        threads=4,
    )
    combined = math.sqrt(
        serial["standard_error"] ** 2
        + parallel["standard_error"] ** 2
    )
    assert abs(serial["price"] - parallel["price"]) < 3 * combined


def test_cpp_surface_prices_match_python():
    from datetime import date, timedelta

    from src.volatility_surface.surface import SurfacePoint, VolSurface

    as_of = date(2026, 9, 4)
    surface = VolSurface(
        as_of=as_of,
        source="test",
        points=[
            SurfacePoint(
                expiry=as_of + timedelta(days=d),
                time_to_expiry=d / 365,
                parameters={
                    "a": 0.03,
                    "b": 0.1,
                    "rho": -0.2,
                    "m": 0.0,
                    "sigma": 0.1,
                },
            )
            for d in (30, 60)
        ],
    )
    out = cpp_surface_prices(
        surface,
        spot=100.0,
        risk_free_rate=0.04,
        moneyness_grid=[-0.05, 0.0, 0.05],
    )
    for i, row in enumerate(out["surface_points"]):
        reference = option_price(
            spot=row["spot"],
            strike=row["strike"],
            time_to_expiry=row["t"],
            risk_free_rate=0.04,
            volatility=surface.interpolate_iv(
                math.log(row["strike"] / row["spot"]), row["t"]
            ),
            option_type="call",
            dividend_yield=0.0,
        )
        assert abs(out["price"][i] - reference) < 1e-9
