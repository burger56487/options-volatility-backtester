"""Property-based tests for pricing invariants."""

import math

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pricing.black_scholes import option_price


@settings(max_examples=60, deadline=None)
@given(
    spot=st.floats(min_value=50, max_value=200),
    strike=st.floats(min_value=50, max_value=200),
    time_to_expiry=st.floats(min_value=0.05, max_value=2.0),
    volatility=st.floats(min_value=0.05, max_value=1.0),
)
def test_european_no_arbitrage_bounds(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
) -> None:
    """European call/put prices respect no-arbitrage bounds."""
    discount_spot = spot * math.exp(-0.01 * time_to_expiry)
    discount_strike = strike * math.exp(-0.04 * time_to_expiry)
    call = option_price(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=0.04,
        volatility=volatility,
        option_type="call",
        dividend_yield=0.01,
    )
    put = option_price(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=0.04,
        volatility=volatility,
        option_type="put",
        dividend_yield=0.01,
    )
    assert call >= max(discount_spot - discount_strike, 0.0) - 1e-9
    assert call <= discount_spot + 1e-9
    assert put >= max(discount_strike - discount_spot, 0.0) - 1e-9
    assert put <= discount_strike + 1e-9


@settings(max_examples=60, deadline=None)
@given(
    spot=st.floats(min_value=50, max_value=200),
    strike=st.floats(min_value=50, max_value=200),
    time_to_expiry=st.floats(min_value=0.05, max_value=2.0),
    volatility=st.floats(min_value=0.05, max_value=1.0),
)
def test_put_call_parity(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
) -> None:
    """European put-call parity holds to tight tolerance."""
    call = option_price(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=0.04,
        volatility=volatility,
        option_type="call",
        dividend_yield=0.01,
    )
    put = option_price(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=0.04,
        volatility=volatility,
        option_type="put",
        dividend_yield=0.01,
    )
    forward_parity = (
        call
        - put
        - (
            spot * math.exp(-0.01 * time_to_expiry)
            - strike * math.exp(-0.04 * time_to_expiry)
        )
    )
    assert abs(forward_parity) < 1e-7
