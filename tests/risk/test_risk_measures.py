import numpy as np
import pandas as pd
import pytest

from src.risk.backtest import (
    christoffersen_independence,
    kupiec_unconditional_coverage,
)
from src.risk.contributions import linear_risk_contributions
from src.risk.measure import (
    delta_gamma_var,
    historical_var,
    liquidity_adjusted_var,
)
from src.risk.scenarios import preset_scenario_shocks, stress_loss


def test_historical_var_requires_sample():
    small = pd.Series(np.full(5, -1.0))
    result = historical_var(small)
    assert result.insufficient_sample is True


def test_delta_gamma_reduces_long_gamma_var():
    plain = delta_gamma_var(50.0, gamma=0.0, spot=100.0, daily_volatility=0.01)
    with_gamma = delta_gamma_var(
        50.0, gamma=10.0, spot=100.0, daily_volatility=0.01
    )
    assert with_gamma.var < plain.var


def test_euler_contributions_sum_to_var():
    exposures = np.array([100.0, -50.0])
    covariance = np.array([[0.01, 0.0], [0.0, 0.04]])
    result = linear_risk_contributions(exposures, covariance)
    assert result["identity_passed"] is True


def test_stress_scenarios():
    exposures = {"equity": 100.0, "volatility": 200.0}
    loss = stress_loss(exposures, preset_scenario_shocks("crash"))
    assert loss["total"] < 0


def test_liquidity_adjusted_var_adds_exit_cost():
    base = liquidity_adjusted_var(
        market_var=10.0,
        position_notional=1000.0,
        relative_half_spread=0.0,
    )
    assert base == 10.0
    with_spread = liquidity_adjusted_var(
        market_var=10.0,
        position_notional=1000.0,
        relative_half_spread=0.01,
    )
    with_impact = liquidity_adjusted_var(
        market_var=10.0,
        position_notional=1000.0,
        relative_half_spread=0.01,
        impact_coefficient=0.005,
        liquidation_fraction=0.5,
    )
    assert with_spread > base
    assert with_impact > with_spread


def test_liquidity_adjusted_var_rejects_invalid_input():
    with pytest.raises(ValueError):
        liquidity_adjusted_var(
            market_var=-1.0,
            position_notional=100.0,
            relative_half_spread=0.01,
        )
    with pytest.raises(ValueError):
        liquidity_adjusted_var(
            market_var=1.0,
            position_notional=100.0,
            relative_half_spread=0.01,
            liquidation_fraction=1.2,
        )


def test_extended_preset_scenarios_are_available():
    for kind in (
        "liquidity_widening",
        "rate_up_100",
        "correlation_up",
        "skew_flattening",
    ):
        shocks = preset_scenario_shocks(kind)
        assert isinstance(shocks, dict) and shocks


def test_backtest_functions_return_p_values():
    rng = np.random.default_rng(4)
    pnl = rng.normal(0, 100, 300)
    var = np.full(300, 1.65 * 100)
    kupiec = kupiec_unconditional_coverage(pnl, var)
    christoffersen = christoffersen_independence(pnl, var)
    assert 0.0 <= kupiec["p_value"] <= 1.0
    assert 0.0 <= christoffersen["p_value"] <= 1.0
