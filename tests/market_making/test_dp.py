"""Tests for the dynamic-programming baseline and closed-form cross-checks."""

from __future__ import annotations

import numpy as np
import pytest

from src.market_making.dp import (
    greedy_action_map,
    mnl_dp_solver,
    exponential_fill_dp_value,
    exponential_fill_closed_form_value,
)
from src.market_making.intensity_env import (
    MMEnvironmentConfig,
    elementary_items,
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


def test_exponential_fill_euler_matches_closed_form():
    """Cross-check against equation (6.4) of the dissertation."""
    cap = 5
    euler = exponential_fill_dp_value(
        phi=5e-3,
        kappa=3.0,
        activity=1.0,
        inventory_cap=cap,
        horizon=15.0,
        n_steps=5_000,
    )
    closed = exponential_fill_closed_form_value(
        phi=5e-3,
        kappa=3.0,
        activity=1.0,
        inventory_cap=cap,
        horizon=15.0,
    )
    assert closed[cap] == pytest.approx(3.42025, abs=5e-4)
    assert euler[0, cap] == pytest.approx(closed[cap], abs=2e-3)
    assert (
        np.abs(euler[0, :] - closed).max()
        < 1e-2
    )


def test_mnl_dp_value_matches_dissertation_benchmark():
    config = _config()
    items = elementary_items(config)
    solver = mnl_dp_solver(
        config=config,
        items=items,
        n_steps=10_000,
    )
    assert solver.values[0, 5] == pytest.approx(1.90299, abs=2e-3)
    assert solver.terminal_zero  # terminal residual value is zero
    assert solver.action_map.shape == (
        solver.n_steps + 1,
        2 * config.inventory_cap + 1,
    )


def test_greedy_action_maximises_immediate_expected_spread():
    config = _config()
    items = elementary_items(config)
    action_map = greedy_action_map(config, items)
    # At zero inventory the greedy policy should never withdraw both sides.
    q_index = config.inventory_cap
    action = action_map[0, q_index]
    assert action != ()


def test_dp_boundary_action_never_breaches_inventory():
    config = _config(inventory_cap=3)
    items = elementary_items(config)
    solver = mnl_dp_solver(
        config=config,
        items=items,
        n_steps=1_000,
    )
    for q in (-3, 3):
        q_index = q + 3
        action = solver.action_map[100, q_index]
        for item_index in action:
            item = items[item_index]
            if item.side == "bid":
                assert q + 1 <= 3
            else:
                assert q - 1 >= -3
