"""Tests for the event-driven single-asset market-making environment."""

from __future__ import annotations

import numpy as np
import pytest

from src.market_making.intensity_env import (
    MMEnvironmentConfig,
    elementary_items,
    feasible_portfolios,
    portfolio_mnl_probabilities,
    run_episode,
    run_grid_episode,
)


def _config(**overrides) -> MMEnvironmentConfig:
    values = dict(
        arrival_rate=0.9,
        kappa=3.0,
        outside_weight=1.0,
        half_spreads=(0.3, 0.6),
        inventory_cap=5,
        horizon=15.0,
        inventory_penalty=5e-3,
    )
    values.update(overrides)
    return MMEnvironmentConfig(**values)


class _TakeAllPolicy:
    """Post the tightest spread on both sides when feasible."""

    def act(self, q: int, t: float):
        items = elementary_items(_config())
        portfolio = []
        for index, item in enumerate(items):
            if item.side == "bid" and q + 1 <= 5:
                portfolio.append(index)
            elif item.side == "ask" and q - 1 >= -5:
                portfolio.append(index)
        return tuple(portfolio)


def test_elementary_items_count_and_order():
    items = elementary_items(_config())
    assert len(items) == 4
    assert items[0].side == "bid"
    assert items[0].half_spread == 0.3
    assert items[1].half_spread == 0.6
    assert items[2].side == "ask"


def test_feasible_portfolio_counts_respect_inventory_cap():
    config = _config()
    items = elementary_items(config)
    interior = feasible_portfolios(0, items, config)
    assert len(interior) == 9
    at_top = feasible_portfolios(5, items, config)
    assert all(
        not any(items[i].side == "bid" for i in portfolio)
        for portfolio in at_top
    )
    assert len(at_top) == 3


def test_mnl_probabilities_sum_to_one_with_outside():
    config = _config()
    items = elementary_items(config)
    portfolio = (0, 3)  # bid tight and ask wide
    probabilities = portfolio_mnl_probabilities(
        portfolio,
        items,
        config,
    )
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[3]


def test_empty_portfolio_falls_through_to_outside():
    config = _config()
    items = elementary_items(config)
    probabilities = portfolio_mnl_probabilities((), items, config)
    assert probabilities[-1] == pytest.approx(1.0)


def test_event_driven_episode_is_deterministic_and_respects_bounds():
    policy = _TakeAllPolicy()
    first = run_episode(policy, _config(), np.random.default_rng(4))
    second = run_episode(policy, _config(), np.random.default_rng(4))
    assert first.net_objective == pytest.approx(second.net_objective)
    assert first.net_objective == pytest.approx(
        first.spread_total - first.penalty_total
    )
    assert first.penalty_total > 0.0
    assert first.n_arrivals > 0
    for fill in first.fills:
        assert -5 <= fill.q_after <= 5
        assert fill.spread in (0.3, 0.6)


def test_same_flow_rng_means_same_arrivals_for_any_policy():
    class _IdlePolicy:
        def act(self, q: int, t: float):
            return ()

    active = run_episode(
        _TakeAllPolicy(),
        _config(),
        np.random.default_rng(11),
    )
    idle = run_episode(
        _IdlePolicy(),
        _config(),
        np.random.default_rng(11),
    )
    assert active.n_arrivals == idle.n_arrivals
    assert active.terminal_time == pytest.approx(idle.terminal_time)
    assert idle.fill_count == 0
    assert active.fill_count > 0


def test_grid_episode_objective_decomposition():
    policy = _TakeAllPolicy()
    result = run_grid_episode(
        policy,
        _config(horizon=12.0),
        dt=1.0,
        rng=np.random.default_rng(9),
    )
    assert result.net_objective == pytest.approx(
        result.spread_total - result.penalty_total
    )
    assert result.grid_steps > 0
    assert np.isfinite(result.net_objective)


def test_fill_never_exceeds_posted_item():
    policy = _TakeAllPolicy()
    result = run_episode(policy, _config(), np.random.default_rng(21))
    for fill in result.fills:
        assert fill.spread in (0.3, 0.6)
