"""Tests for the Linear-MC and discretised-time Actor-Critic learners."""

from __future__ import annotations

import numpy as np
import pytest

from src.market_making.intensity_env import (
    MMEnvironmentConfig,
    elementary_items,
    feasible_portfolios,
)
from src.market_making.rl import (
    LinearCritic,
    PairwiseActor,
    critic_basis,
    train_discrete_ac,
    train_linear_mc,
)


def _config(**overrides) -> MMEnvironmentConfig:
    values = dict(
        arrival_rate=1.0,
        kappa=3.0,
        outside_weight=1.0,
        half_spreads=(0.4, 0.8),
        inventory_cap=3,
        horizon=12.0,
        inventory_penalty=5e-3,
    )
    values.update(overrides)
    return MMEnvironmentConfig(**values)


def test_critic_basis_vanishes_at_terminal():
    basis = critic_basis(t=12.0, q=2, config=_config())
    assert len(basis) == 6
    assert np.allclose(basis, 0.0)
    interior = critic_basis(t=5.0, q=1, config=_config())
    assert np.isfinite(interior).all()


def test_linear_critic_recovers_known_function():
    config = _config()
    rng = np.random.default_rng(0)
    n = 120
    times = rng.uniform(1.0, config.horizon - 1.0, n)
    qs = rng.integers(-config.inventory_cap, config.inventory_cap + 1, n)
    features = np.array(
        [critic_basis(float(t), int(q), config) for t, q in zip(times, qs)]
    )
    true_w = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.2])
    targets = features @ true_w
    critic = LinearCritic(config)
    critic.fit(features, targets, ridge=0.0)
    assert np.allclose(critic.weights, true_w, atol=1e-6)


def test_actor_softmax_over_feasible_portfolios():
    config = _config()
    items = elementary_items(config)
    actor = PairwiseActor(config, items, seed=1)
    for q in (-3, -1, 0, 2, 3):
        probabilities = actor.probabilities(q)
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert set(probabilities) == {
            p
            for p in feasible_portfolios(q, items, config)
        }


def test_actor_score_gradient_matches_finite_difference():
    config = _config()
    items = elementary_items(config)
    actor = PairwiseActor(config, items, seed=2)
    q = 1
    portfolio = feasible_portfolios(q, items, config)[3]
    gradient = actor.score_gradient(q, portfolio)
    step = 1e-6
    base = actor.log_probability(q, portfolio)
    i, j = 2, 1
    actor.parameters[i, j] += step
    bumped = actor.log_probability(q, portfolio)
    actor.parameters[i, j] -= step
    numeric = (bumped - base) / step
    assert gradient[i, j] == pytest.approx(numeric, rel=1e-3, abs=1e-5)


def test_linear_mc_training_improves_over_uniform():
    config = _config()
    actor = train_linear_mc(
        config=config,
        episodes=400,
        batch_size=20,
        seed=3,
    )
    trained = _evaluate_mean(actor, config, seed=101, episodes=120)
    uniform = _evaluate_mean(_UniformPolicy(config), config, seed=101, episodes=120)
    assert trained["fill_count"] > 0
    assert trained["net_objective"] > uniform["net_objective"] - 0.2


def test_discrete_ac_trainer_runs_and_returns_actor():
    config = _config(horizon=10.0)
    actor = train_discrete_ac(
        config=config,
        episodes=60,
        batch_size=20,
        dt=1.0,
        seed=4,
    )
    probabilities = actor.probabilities(0)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def _evaluate_mean(policy, config, seed: int, episodes: int) -> dict:
    from src.market_making.intensity_env import run_episode

    objectives = []
    fill_counts = []
    for episode in range(episodes):
        rng = np.random.default_rng(seed * 10_000 + episode)
        result = run_episode(policy, config, rng)
        objectives.append(result.net_objective)
        fill_counts.append(result.fill_count)
    return {
        "net_objective": float(np.mean(objectives)),
        "fill_count": float(np.mean(fill_counts)),
    }


class _UniformPolicy:
    def __init__(self, config: MMEnvironmentConfig) -> None:
        self._items = elementary_items(config)
        self._config = config
        self._rng = np.random.default_rng(0)

    def act(self, q: int, t: float):
        portfolios = feasible_portfolios(q, self._items, self._config)
        return portfolios[int(self._rng.integers(len(portfolios)))]
