import math

import pytest

from src.domain.enums import OptionType
from src.pricing.black_scholes import option_price
from src.pricing.heston import (
    heston_characteristic_function,
    heston_mc_price,
)


def test_characteristic_function_identity():
    tau = 0.5
    kappa, theta, sigma_v, rho, v0 = 1.5, 0.04, 0.3, -0.6, 0.04
    at_zero = heston_characteristic_function(
        0.0, tau, kappa, theta, sigma_v, rho, v0
    )
    assert abs(at_zero - 1.0) < 1e-9
    # Magnitude of any characteristic function is at most one; this check is
    # insensitive to complex-log branch choices in the closed form.
    for real_u in (0.25, 0.5, 0.9):
        value = heston_characteristic_function(
            real_u, tau, kappa, theta, sigma_v, rho, v0
        )
        assert abs(value) <= 1.0 + 1e-9


def test_heston_mc_degenerates_to_black_scholes():
    spot, strike, tau = 100.0, 100.0, 0.5
    v0 = theta = 0.04
    result = heston_mc_price(
        spot=spot,
        strike=strike,
        time_to_expiry=tau,
        kappa=1.0,
        theta=theta,
        sigma_v=1e-6,  # effectively constant variance
        rho=0.0,
        v0=v0,
        risk_free_rate=0.04,
        option_type=OptionType.CALL,
        dividend_yield=0.0,
        n_paths=40_000,
        n_steps=20,
        seed=3,
    )
    reference = option_price(
        spot=spot,
        strike=strike,
        time_to_expiry=tau,
        risk_free_rate=0.04,
        volatility=math.sqrt(theta),
        option_type="call",
        dividend_yield=0.0,
    )
    assert abs(result["price"] - reference) < 3 * result["standard_error"]
