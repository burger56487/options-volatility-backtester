import numpy as np

from src.hedging.evaluate import compare_policies, simulate_hedge
from src.hedging.policies import (
    FixedDeltaHedge,
    ThresholdDeltaHedge,
    cost_aware_band,
    neutralize_delta_gamma,
)
from src.hedging.rl_env import HedgeEnv, train_linear_policy


def test_cost_aware_band_increases_with_cost():
    low = cost_aware_band(0.1, 100.0, 0.01, 0.2, 1 / 252)
    high = cost_aware_band(0.1, 100.0, 0.05, 0.2, 1 / 252)
    assert high > low


def test_delta_gamma_neutralisation_clears_exposure():
    stock_shares, option_qty = neutralize_delta_gamma(
        delta_portfolio=50.0,
        gamma_portfolio=5.0,
        delta_secondary_option=0.5 * 100,
        gamma_secondary_option=0.01 * 100,
    )
    residual_delta = 50.0 + stock_shares + option_qty * (0.5 * 100)
    residual_gamma = 5.0 + option_qty * (0.01 * 100)
    assert abs(residual_delta) < 1e-6
    assert abs(residual_gamma) < 1e-6


def test_threshold_hedge_trades_less_than_fixed():
    rng = np.random.default_rng(3)
    spot = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 60)))
    delta = 50.0 * np.ones(60) + 5.0 * np.sin(np.arange(60) / 5)
    comparison = compare_policies(spot, delta, cost_bps=1.0)
    assert (
        comparison["threshold"]["trade_count"]
        < comparison["fixed"]["trade_count"]
    )
    assert (
        comparison["threshold"]["total_cost"]
        < comparison["fixed"]["total_cost"]
    )


def test_rl_env_and_training_smoke():
    rng = np.random.default_rng(1)
    spot = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, 80)))
    delta = 40.0 + 10.0 * np.sin(np.arange(80) / 4)
    env = HedgeEnv(spot, delta)
    weight = train_linear_policy(env, episodes=10, seed=2)
    assert np.isfinite(weight)
    state = env.reset()
    assert state.shape == (3,)
