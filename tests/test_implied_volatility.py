import pytest

from src.pricing.black_scholes import option_price
from src.pricing.implied_volatility import (
    implied_volatility,
    implied_volatility_bisection,
    implied_volatility_newton,
)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_bisection_recovers_known_volatility(
    option_type: str,
):
    true_volatility = 0.28

    market_price = option_price(
        spot=100.0,
        strike=105.0,
        time_to_expiry=0.75,
        risk_free_rate=0.03,
        volatility=true_volatility,
        option_type=option_type,  # type: ignore
        dividend_yield=0.01,
    )

    recovered_volatility = implied_volatility_bisection(
        market_price=market_price,
        spot=100.0,
        strike=105.0,
        time_to_expiry=0.75,
        risk_free_rate=0.03,
        option_type=option_type,  # type: ignore
        dividend_yield=0.01,
    )

    assert recovered_volatility == pytest.approx(
        true_volatility,
        abs=1e-6,
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_newton_recovers_known_volatility(
    option_type: str,
):
    true_volatility = 0.35

    market_price = option_price(
        spot=120.0,
        strike=110.0,
        time_to_expiry=1.25,
        risk_free_rate=0.04,
        volatility=true_volatility,
        option_type=option_type,  # type: ignore
    )

    recovered_volatility = implied_volatility_newton(
        market_price=market_price,
        spot=120.0,
        strike=110.0,
        time_to_expiry=1.25,
        risk_free_rate=0.04,
        option_type=option_type,  # type: ignore
        initial_volatility=0.15,
    )

    assert recovered_volatility == pytest.approx(
        true_volatility,
        abs=1e-6,
    )


def test_default_solver_recovers_known_volatility():
    true_volatility = 0.22

    market_price = option_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.50,
        risk_free_rate=0.05,
        volatility=true_volatility,
        option_type="call",
    )

    recovered_volatility = implied_volatility(
        market_price=market_price,
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.50,
        risk_free_rate=0.05,
        option_type="call",
    )

    assert recovered_volatility == pytest.approx(
        true_volatility,
        abs=1e-6,
    )


def test_price_below_no_arbitrage_bound_raises_error():
    with pytest.raises(
        ValueError,
        match="below the no-arbitrage lower bound",
    ):
        implied_volatility(
            market_price=0.01,
            spot=100.0,
            strike=50.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            option_type="call",
        )


def test_price_above_no_arbitrage_bound_raises_error():
    with pytest.raises(
        ValueError,
        match="above the no-arbitrage upper bound",
    ):
        implied_volatility(
            market_price=120.0,
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            option_type="call",
        )


def test_negative_market_price_raises_error():
    with pytest.raises(
        ValueError,
        match="market_price must be non-negative",
    ):
        implied_volatility(
            market_price=-1.0,
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            option_type="call",
        )


def test_invalid_solver_method_raises_error():
    with pytest.raises(
        ValueError,
        match="method must be either",
    ):
        implied_volatility(
            market_price=10.0,
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            option_type="call",
            method="brent",  # type: ignore
        )


def test_newton_can_fall_back_to_bisection():
    true_volatility = 0.30

    market_price = option_price(
        spot=100.0,
        strike=150.0,
        time_to_expiry=0.10,
        risk_free_rate=0.03,
        volatility=true_volatility,
        option_type="call",
    )

    recovered_volatility = implied_volatility_newton(
        market_price=market_price,
        spot=100.0,
        strike=150.0,
        time_to_expiry=0.10,
        risk_free_rate=0.03,
        option_type="call",
        initial_volatility=0.01,
        minimum_vega=1e-4,
    )

    assert recovered_volatility == pytest.approx(
    true_volatility,
    abs=1e-5,
)

