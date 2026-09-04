import math

import pytest

from src.domain.enums import ExerciseStyle, OptionType
from src.numerics.tridiagonal import solve_tridiagonal
from src.pricing.binomial import crr_price
from src.pricing.black_scholes import option_price
from src.pricing.finite_difference import finite_difference_price
from src.pricing.requests import PricingRequest


def _request(
    option_type: OptionType,
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN,
    steps: int = 300,
) -> PricingRequest:
    return PricingRequest(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        volatility=0.25,
        option_type=option_type,
        exercise_style=exercise_style,
        steps=steps,
    )


def test_tridiagonal_solver_matches_known_solution():
    solution = solve_tridiagonal(
        lower=[1.0, 1.0],
        diagonal=[2.0, 2.0, 2.0],
        upper=[1.0, 1.0],
        rhs=[4.0, 6.0, 8.0],
    )
    assert solution == pytest.approx([2.0, 0.0, 4.0], abs=1e-9)


def test_crr_converges_to_black_scholes():
    for option_type in (OptionType.CALL, OptionType.PUT):
        reference = option_price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.5,
            risk_free_rate=0.04,
            volatility=0.25,
            option_type=option_type.value,
            dividend_yield=0.01,
        )
        tree = crr_price(_request(option_type, steps=400))
        assert tree.price == pytest.approx(reference, abs=0.02)


def test_crank_nicolson_matches_black_scholes():
    request = _request(OptionType.CALL)
    reference = option_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        volatility=0.25,
        option_type="call",
        dividend_yield=0.01,
    )
    fd = finite_difference_price(
        request,
        theta=0.5,
        x_steps=600,
        time_steps=300,
    )
    assert fd.price == pytest.approx(reference, abs=0.06)


def test_explicit_scheme_rejects_unstable_grid():
    request = _request(OptionType.CALL)
    with pytest.raises(ValueError, match="stability"):
        finite_difference_price(
            request,
            theta=0.0,
            x_steps=100,
            time_steps=10,
        )


def test_american_put_exceeds_european_price():
    european = crr_price(
        _request(OptionType.PUT, ExerciseStyle.EUROPEAN, steps=300)
    )
    american = crr_price(
        _request(OptionType.PUT, ExerciseStyle.AMERICAN, steps=300)
    )
    assert american.price >= european.price - 1e-12
