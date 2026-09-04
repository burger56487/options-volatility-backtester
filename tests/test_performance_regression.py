"""Performance regression: the C++ backend must stay far above Python.

Tests are marked ``perf`` because wall-clock measurements are
environment-sensitive; the default suite and CI exclude this marker. The
floor is deliberately conservative (2x) so shared runners do not flake; the
measured ratio on the 2026-09-04 baseline machine was ~3.8x.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from src.pricing.cpp_backend import cpp_batch_bs, is_available

pytestmark = pytest.mark.perf

N_PATHS = 1_000_000


def _bs_call(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    sigma_sqrt_t = volatility * math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2)
        * time_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return (
        spot
        * math.exp(-dividend_yield * time_to_expiry)
        * (0.5 * math.erfc(-d1 / math.sqrt(2.0)))
        - strike
        * math.exp(-risk_free_rate * time_to_expiry)
        * (0.5 * math.erfc(-d2 / math.sqrt(2.0)))
    )


@pytest.mark.perf
def test_cpp_batch_bs_speedup_above_floor() -> None:
    if not is_available():
        pytest.skip("C++ backend not built")
    spot = strike = 100.0
    time_to_expiry, risk_free_rate = 0.5, 0.04
    dividend_yield, volatility = 0.01, 0.25

    start = time.perf_counter()
    total = 0.0
    for _ in range(N_PATHS):
        total += _bs_call(
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            dividend_yield,
            volatility,
        )
    python_seconds = time.perf_counter() - start
    python_average = total / N_PATHS

    start = time.perf_counter()
    output = cpp_batch_bs(
        np.full(N_PATHS, spot),
        np.full(N_PATHS, strike),
        np.full(N_PATHS, time_to_expiry),
        risk_free_rate,
        dividend_yield,
        np.full(N_PATHS, volatility),
        np.array(["call"] * N_PATHS),
    )
    cpp_seconds = time.perf_counter() - start

    assert abs(python_average - float(output["price"][0])) < 1e-8
    speedup = python_seconds / cpp_seconds
    assert speedup > 2.0, f"speedup {speedup:.1f}x fell below 2x floor"
