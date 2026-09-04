"""Tests for the fair RL-vs-DP policy comparison study."""

from __future__ import annotations

import pandas as pd
import pytest

from src.market_making.dp import mnl_dp_solver
from src.market_making.intensity_env import (
    MMEnvironmentConfig,
    elementary_items,
)
from src.market_making.study import (
    DPPolicy,
    GreedyPolicy,
    UniformPolicy,
    evaluate_policy_episodes,
    run_policy_comparison,
    save_study_results,
    train_rl_policy_for_study,
)


def _config(**overrides) -> MMEnvironmentConfig:
    values = dict(
        arrival_rate=0.9,
        kappa=3.0,
        outside_weight=1.0,
        half_spreads=(0.3, 0.6),
        inventory_cap=3,
        horizon=8.0,
        inventory_penalty=5e-3,
    )
    values.update(overrides)
    return MMEnvironmentConfig(**values)


def test_evaluation_shares_the_same_order_flow_for_a_seed():
    config = _config()
    greedy = GreedyPolicy(config)
    greedy_eval = evaluate_policy_episodes(greedy, config, seed=7, episodes=12)
    uniform = UniformPolicy(config)
    uniform_eval = evaluate_policy_episodes(
        uniform,
        config,
        seed=7,
        episodes=12,
    )
    assert (greedy_eval["n_arrivals"] == uniform_eval["n_arrivals"]).all()


def test_tiny_comparison_produces_rows_and_aggregation():
    config = _config(horizon=6.0)
    result = run_policy_comparison(
        config=config,
        seeds=(1, 2),
        n_eval_episodes=8,
        dp_steps=200,
        rl_episodes=60,
        rl_batch_size=15,
    )
    assert isinstance(result["rows"], pd.DataFrame)
    policies = sorted(result["rows"]["policy"].unique())
    assert policies == ["dp", "greedy", "rl_linear_mc", "uniform"]
    assert (result["rows"].groupby(["policy", "seed"]).size() > 0).all()
    aggregated = result["aggregated"]
    assert len(aggregated) == 4
    assert (aggregated["ci_low"] <= aggregated["ci_high"]).all()


def test_training_helper_returns_actor():
    config = _config(horizon=6.0)
    actor = train_rl_policy_for_study(
        config=config,
        episodes=50,
        batch_size=10,
        seed=5,
    )
    probabilities = actor.probabilities(0)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_save_study_results_writes_outputs(tmp_path):
    config = _config(horizon=5.0)
    result = run_policy_comparison(
        config=config,
        seeds=(1,),
        n_eval_episodes=5,
        dp_steps=100,
        rl_episodes=40,
        rl_batch_size=10,
    )
    output = save_study_results(result, tmp_path)
    assert (output / "comparison.csv").exists()
    assert (output / "aggregated.csv").exists()
    assert (output / "summary.json").exists()
