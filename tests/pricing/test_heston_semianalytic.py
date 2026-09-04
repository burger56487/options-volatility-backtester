"""Semi-analytic Heston pricing and parameter calibration."""

from __future__ import annotations

import math

import numpy as np

from src.domain.enums import OptionType
from src.pricing.black_scholes import option_price
from src.pricing.heston import (
    _simulate_heston_paths,
    heston_characteristic_function,
    heston_mc_price,
    heston_price_semi_analytic,
    heston_prices_semi_analytic,
)
from src.pricing.heston_calibration import (
    HestonQuote,
    calibrate_heston,
)


def _params() -> dict:
    return dict(
        spot=100.0,
        kappa=1.0,
        theta=0.04,
        sigma_v=0.25,
        rho=-0.4,
        v0=0.04,
        risk_free_rate=0.04,
        dividend_yield=0.01,
    )


def test_semi_analytic_degenerates_to_black_scholes() -> None:
    params = _params()
    params.update(sigma_v=1e-9, rho=0.0)
    reference = option_price(
        spot=params["spot"],
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=params["risk_free_rate"],
        volatility=math.sqrt(params["theta"]),
        option_type="call",
        dividend_yield=params["dividend_yield"],
    )
    price = heston_price_semi_analytic(
        spot=params["spot"],
        strike=100.0,
        time_to_expiry=0.5,
        kappa=params["kappa"],
        theta=params["theta"],
        sigma_v=params["sigma_v"],
        rho=params["rho"],
        v0=params["v0"],
        risk_free_rate=params["risk_free_rate"],
        option_type=OptionType.CALL,
        dividend_yield=params["dividend_yield"],
    )
    assert abs(price - reference) < 1e-9


def test_semi_analytic_matches_monte_carlo() -> None:
    params = _params()
    semi = heston_price_semi_analytic(
        spot=params["spot"],
        strike=100.0,
        time_to_expiry=0.5,
        kappa=params["kappa"],
        theta=params["theta"],
        sigma_v=params["sigma_v"],
        rho=params["rho"],
        v0=params["v0"],
        risk_free_rate=params["risk_free_rate"],
        option_type=OptionType.CALL,
        dividend_yield=params["dividend_yield"],
    )
    mc = heston_mc_price(
        spot=params["spot"],
        strike=100.0,
        time_to_expiry=0.5,
        kappa=params["kappa"],
        theta=params["theta"],
        sigma_v=params["sigma_v"],
        rho=params["rho"],
        v0=params["v0"],
        risk_free_rate=params["risk_free_rate"],
        option_type=OptionType.CALL,
        dividend_yield=params["dividend_yield"],
        n_paths=200_000,
        n_steps=60,
        seed=5,
    )
    assert abs(semi - mc["price"]) < 3 * mc["standard_error"]


def test_semi_analytic_put_call_parity() -> None:
    params = _params()
    tau = 0.5
    call = heston_price_semi_analytic(
        spot=params["spot"],
        strike=100.0,
        time_to_expiry=tau,
        kappa=params["kappa"],
        theta=params["theta"],
        sigma_v=params["sigma_v"],
        rho=params["rho"],
        v0=params["v0"],
        risk_free_rate=params["risk_free_rate"],
        option_type=OptionType.CALL,
        dividend_yield=params["dividend_yield"],
    )
    put = heston_price_semi_analytic(
        spot=params["spot"],
        strike=100.0,
        time_to_expiry=tau,
        kappa=params["kappa"],
        theta=params["theta"],
        sigma_v=params["sigma_v"],
        rho=params["rho"],
        v0=params["v0"],
        risk_free_rate=params["risk_free_rate"],
        option_type=OptionType.PUT,
        dividend_yield=params["dividend_yield"],
    )
    forward = (
        params["spot"] * math.exp(-params["dividend_yield"] * tau)
        - 100.0 * math.exp(-params["risk_free_rate"] * tau)
    )
    assert abs((call - put) - forward) < 1e-6


def test_characteristic_function_matches_simulated_distribution() -> None:
    params = _params()
    tau = 0.5
    terminal, _ = _simulate_heston_paths(
        spot=params["spot"],
        time_to_expiry=tau,
        kappa=params["kappa"],
        theta=params["theta"],
        sigma_v=params["sigma_v"],
        rho=params["rho"],
        v0=params["v0"],
        risk_free_rate=0.0,
        dividend_yield=0.0,
        n_paths=400_000,
        n_steps=50,
        seed=7,
    )
    log_return = np.log(terminal / params["spot"])
    for u in (0.5, 1.5, 2.5):
        mc_estimate = complex(np.mean(np.exp(1j * u * log_return)))
        cf = heston_characteristic_function(
            u,
            tau,
            params["kappa"],
            params["theta"],
            params["sigma_v"],
            params["rho"],
            params["v0"],
        )
        assert abs(mc_estimate - cf) < 0.015


def test_heston_calibration_recovers_synthetic_parameters() -> None:
    params = _params()
    quotes = []
    for tau in (0.5, 1.0):
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
        prices = heston_prices_semi_analytic(
            spot=params["spot"],
            strikes=strikes,
            time_to_expiries=np.full(len(strikes), tau),
            option_types=[OptionType.CALL] * len(strikes),
            kappa=params["kappa"],
            theta=params["theta"],
            sigma_v=params["sigma_v"],
            rho=params["rho"],
            v0=params["v0"],
            risk_free_rate=params["risk_free_rate"],
            dividend_yield=params["dividend_yield"],
        )
        for strike, price in zip(strikes, prices):
            quotes.append(
                HestonQuote(
                    strike=float(strike),
                    time_to_expiry=float(tau),
                    option_type=OptionType.CALL,
                    market_price=float(price),
                )
            )
    result = calibrate_heston(
        spot=params["spot"],
        quotes=quotes,
        risk_free_rate=params["risk_free_rate"],
        dividend_yield=params["dividend_yield"],
        initial_guesses=(
            (1.0, 0.04, 0.3, -0.5, 0.04),
            (0.5, 0.06, 0.4, -0.2, 0.06),
        ),
        max_iterations=120,
        grid_points=3_000,
    )
    assert result.converged
    assert result.rmse < 1e-2
    assert abs(result.parameters["kappa"] - params["kappa"]) < 0.15
    assert abs(result.parameters["theta"] - params["theta"]) < 0.005
    assert abs(result.parameters["sigma_v"] - params["sigma_v"]) < 0.03
    assert abs(result.parameters["rho"] - params["rho"]) < 0.03
    assert abs(result.parameters["v0"] - params["v0"]) < 0.005
