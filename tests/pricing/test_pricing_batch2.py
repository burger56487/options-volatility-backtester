import pytest

from src.domain.enums import OptionType
from src.pricing.black_scholes import option_price
from src.pricing.convergence import convergence_study, crr_with_steps
from src.pricing.greeks import greeks_consistency_report
from src.pricing.monte_carlo import monte_carlo_price
from src.pricing.registry import available_methods, price
from src.pricing.requests import PricingRequest


def _request(steps: int = 400) -> PricingRequest:
    return PricingRequest(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        volatility=0.25,
        option_type=OptionType.CALL,
        steps=steps,
    )


def _reference() -> float:
    return option_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        volatility=0.25,
        option_type="call",
        dividend_yield=0.01,
    )


def test_monte_carlo_ci_covers_analytic_and_reduces_variance():
    mc = monte_carlo_price(_request(), n_paths=80_000, seed=11)
    reference = _reference()
    assert abs(mc["price"] - reference) < 3 * mc["standard_error"]
    assert mc["variance_reduction_ratio"] > 1.0


def test_registry_dispatches_all_methods():
    reference = _reference()
    tolerances = {
        "black_scholes": 1e-9,
        "crr": 0.02,
        "crank_nicolson": 0.06,
        "monte_carlo": 0.25,
    }
    assert set(tolerances) == set(available_methods())
    for method, tolerance in tolerances.items():
        result = price(_request(), method)
        assert abs(result.price - reference) < tolerance


def test_greeks_consistency_with_analytic():
    report = greeks_consistency_report(_request())
    assert report["delta_diff"] < 2e-3
    assert report["gamma_diff"] < 5e-2
    assert report["vega_diff"] < 0.2
    assert report["theta_diff"] < 5.0


def test_crr_convergence_order_ratio():
    frame = convergence_study(
        _request(),
        _reference(),
        crr_with_steps,
        step_name="steps",
        steps=[100, 200, 400],
    )
    order = frame["observed_order"].dropna()
    assert 1.5 <= float(order.iloc[0]) <= 3.0
    assert 1.5 <= float(order.iloc[1]) <= 3.0
