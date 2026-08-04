import math

import pytest

from src.pricing.black_scholes import (
    option_price,
    price_and_greeks,
)


def test_call_price_matches_known_reference_value():
    result = price_and_greeks(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type="call",
    )

    assert result.price == pytest.approx(
        10.4506,
        abs=1e-4,
    )


def test_put_price_matches_known_reference_value():
    result = price_and_greeks(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type="put",
    )

    assert result.price == pytest.approx(
        5.5735,
        abs=1e-4,
    )


def test_put_call_parity_without_dividends():
    spot = 100.0
    strike = 105.0
    time_to_expiry = 0.75
    risk_free_rate = 0.03
    volatility = 0.25

    call = option_price(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type="call",
    )

    put = option_price(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type="put",
    )

    expected_difference = (
        spot
        - strike * math.exp(-risk_free_rate * time_to_expiry)
    )

    assert call - put == pytest.approx(
        expected_difference,
        abs=1e-10,
    )


def test_put_call_parity_with_dividends():
    spot = 100.0
    strike = 105.0
    time_to_expiry = 0.75
    risk_free_rate = 0.03
    dividend_yield = 0.02
    volatility = 0.25

    call = option_price(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type="call",
        dividend_yield=dividend_yield,
    )

    put = option_price(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        option_type="put",
        dividend_yield=dividend_yield,
    )

    expected_difference = (
        spot * math.exp(-dividend_yield * time_to_expiry)
        - strike * math.exp(-risk_free_rate * time_to_expiry)
    )

    assert call - put == pytest.approx(
        expected_difference,
        abs=1e-10,
    )


def test_call_delta_is_between_zero_and_one():
    result = price_and_greeks(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type="call",
    )

    assert 0.0 < result.delta < 1.0


def test_put_delta_is_between_minus_one_and_zero():
    result = price_and_greeks(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type="put",
    )

    assert -1.0 < result.delta < 0.0


def test_call_and_put_have_equal_gamma_and_vega():
    call = price_and_greeks(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type="call",
    )

    put = price_and_greeks(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
        option_type="put",
    )

    assert call.gamma == pytest.approx(put.gamma)
    assert call.vega == pytest.approx(put.vega)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("spot", 0.0, "spot must be positive"),
        ("strike", 0.0, "strike must be positive"),
        (
            "time_to_expiry",
            0.0,
            "time_to_expiry must be positive",
        ),
        (
            "volatility",
            0.0,
            "volatility must be positive",
        ),
    ],
)
def test_invalid_positive_inputs_raise_errors(
    parameter: str,
    value: float,
    message: str,
):
    inputs = {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "risk_free_rate": 0.05,
        "volatility": 0.20,
        "option_type": "call",
    }

    inputs[parameter] = value

    with pytest.raises(ValueError, match=message):
        option_price(**inputs)


def test_invalid_option_type_raises_error():
    with pytest.raises(
        ValueError,
        match="option_type must be either",
    ):
        option_price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="straddle",  # type: ignore
        )
