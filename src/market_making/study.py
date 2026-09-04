"""Fair policy-comparison study: DP, greedy, uniform and event-driven RL.

All policies are evaluated on the same event-driven simulator.  For a given
seed every policy sees the identical Poisson arrival and MNL-routing stream
(the flow randomness is drawn independently of the policy), which makes the
comparison fair.  Training dispersion is reported across multiple seeds with
confidence intervals, following the multi-seed protocol of the dissertation's
experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dp import (
    DPSolution,
    dp_value_at_zero,
    greedy_action_map,
    mnl_dp_solver,
)
from .intensity_env import (
    MMEnvironmentConfig,
    elementary_items,
    feasible_portfolios,
    run_episode,
)
from .rl import PairwiseActor, train_linear_mc


class DPPolicy:
    """Optimal finite-state policy read from the DP solver's action map."""

    def __init__(
        self,
        solver: DPSolution,
        config: MMEnvironmentConfig,
    ) -> None:
        self.action_map = solver.action_map
        self.n_steps = solver.n_steps
        self.config = config

    def act(self, q: int, t: float) -> tuple[int, ...]:
        index = min(
            int(t / self.config.horizon * self.n_steps),
            self.n_steps,
        )
        q_index = q + self.config.inventory_cap
        return self.action_map[index, q_index]


class GreedyPolicy:
    """Model-based short-sighted policy maximising immediate expected spread."""

    def __init__(self, config: MMEnvironmentConfig) -> None:
        self.config = config
        self.action_map = greedy_action_map(config)

    def act(self, q: int, t: float) -> tuple[int, ...]:
        q_index = q + self.config.inventory_cap
        return self.action_map[0, q_index]


class UniformPolicy:
    """Stationary uniform-random baseline over the feasible portfolios."""

    def __init__(
        self,
        config: MMEnvironmentConfig,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.items = elementary_items(config)
        self.rng = np.random.default_rng(seed)

    def act(self, q: int, t: float) -> tuple[int, ...]:
        portfolios = feasible_portfolios(q, self.items, self.config)
        return portfolios[int(self.rng.integers(len(portfolios)))]


def evaluate_policy_episodes(
    policy,
    config: MMEnvironmentConfig,
    seed: int,
    episodes: int,
) -> pd.DataFrame:
    """Evaluate one policy on ``episodes`` seeded evaluation streams."""
    rows = []
    for episode in range(episodes):
        rng = np.random.default_rng(seed * 10_000 + episode)
        result = run_episode(policy, config, rng)
        rows.append(
            {
                "episode": episode,
                "net_objective": result.net_objective,
                "spread_total": result.spread_total,
                "penalty_total": result.penalty_total,
                "fill_count": result.fill_count,
                "n_arrivals": result.n_arrivals,
                "final_inventory": result.final_inventory,
            }
        )
    return pd.DataFrame(rows)


def train_rl_policy_for_study(
    config: MMEnvironmentConfig,
    episodes: int,
    batch_size: int,
    seed: int,
) -> PairwiseActor:
    """Train the event-driven Linear-MC policy for one comparison seed."""
    return train_linear_mc(
        config=config,
        episodes=episodes,
        batch_size=batch_size,
        seed=seed,
    )


def _summary_from_episodes(frame: pd.DataFrame) -> dict:
    values = frame["net_objective"].to_numpy(dtype=float)
    count = values.size
    mean = float(values.mean()) if count else float("nan")
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    sem = std / np.sqrt(count) if count > 1 else float("nan")
    return {
        "mean_net_objective": mean,
        "mean_spread_total": float(frame["spread_total"].mean()),
        "mean_penalty_total": float(frame["penalty_total"].mean()),
        "mean_fill_count": float(frame["fill_count"].mean()),
        "mean_n_arrivals": float(frame["n_arrivals"].mean()),
        "std_net_objective": std,
        "ci_low": mean - 1.96 * sem if count > 1 else mean,
        "ci_high": mean + 1.96 * sem if count > 1 else mean,
        "n_episodes": count,
    }


def run_policy_comparison(
    config: MMEnvironmentConfig,
    seeds: tuple[int, ...] = (123, 456, 789),
    n_eval_episodes: int = 200,
    dp_steps: int = 10_000,
    rl_episodes: int = 3_000,
    rl_batch_size: int = 20,
) -> dict:
    """Run the multi-seed comparison and return seed- and episode-level frames."""
    items = elementary_items(config)
    solver = mnl_dp_solver(config, items, n_steps=dp_steps)
    dp_policy = DPPolicy(solver, config)
    greedy_policy = GreedyPolicy(config)
    uniform_policy = UniformPolicy(config)

    seed_rows = []
    episode_frames = []
    for seed in seeds:
        policies = {
            "dp": dp_policy,
            "greedy": greedy_policy,
            "uniform": uniform_policy,
            "rl_linear_mc": train_rl_policy_for_study(
                config=config,
                episodes=rl_episodes,
                batch_size=rl_batch_size,
                seed=seed,
            ),
        }
        for policy_name, policy in policies.items():
            frame = evaluate_policy_episodes(
                policy,
                config,
                seed=seed,
                episodes=n_eval_episodes,
            )
            frame.insert(0, "policy", policy_name)
            frame.insert(1, "seed", seed)
            episode_frames.append(frame)
            summary = _summary_from_episodes(frame)
            seed_rows.append(
                {
                    "policy": policy_name,
                    "seed": seed,
                    **summary,
                }
            )
    rows = pd.DataFrame(seed_rows)
    aggregated = _aggregate_seed_rows(rows)
    episode_metrics = pd.concat(episode_frames, ignore_index=True)
    return {
        "rows": rows,
        "aggregated": aggregated,
        "episode_metrics": episode_metrics,
        "dp_value_zero": dp_value_at_zero(solver, config),
        "dp_steps": dp_steps,
        "config": config,
    }


def _aggregate_seed_rows(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for policy, group in rows.groupby("policy", sort=True):
        means = group["mean_net_objective"].to_numpy(dtype=float)
        count = means.size
        mean = float(means.mean())
        std = float(means.std(ddof=1)) if count > 1 else 0.0
        sem = std / np.sqrt(count) if count > 1 else float("nan")
        output.append(
            {
                "policy": policy,
                "n_seeds": count,
                "mean": mean,
                "median": float(np.median(means)),
                "std": std,
                "ci_low": mean - 1.96 * sem if count > 1 else mean,
                "ci_high": mean + 1.96 * sem if count > 1 else mean,
                "min": float(means.min()) if count else float("nan"),
                "max": float(means.max()) if count else float("nan"),
            }
        )
    return pd.DataFrame(output)


def save_study_results(result: dict, output_directory: str | Path) -> Path:
    """Write the comparison tables, episode metrics and a summary JSON."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    result["rows"].to_csv(
        output_path / "comparison.csv",
        index=False,
    )
    result["aggregated"].to_csv(
        output_path / "aggregated.csv",
        index=False,
    )
    result["episode_metrics"].to_csv(
        output_path / "episode_metrics.csv",
        index=False,
    )
    config = result["config"]
    summary = {
        "dp_value_zero": result["dp_value_zero"],
        "dp_steps": result["dp_steps"],
        "environment": {
            "arrival_rate": config.arrival_rate,
            "kappa": config.kappa,
            "outside_weight": config.outside_weight,
            "half_spreads": list(config.half_spreads),
            "inventory_cap": config.inventory_cap,
            "horizon": config.horizon,
            "inventory_penalty": config.inventory_penalty,
        },
        "disclaimer": (
            "Research simulation only. Option quotes and fills are synthetic; "
            "results do not represent live trading or a real venue."
        ),
    }
    with (output_path / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    _plot_comparison(result, output_path)
    return output_path


def _plot_comparison(result: dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregated = result["aggregated"]
    order = ["dp", "greedy", "uniform", "rl_linear_mc"]
    present = [
        policy for policy in order if policy in set(aggregated["policy"])
    ]
    grouped = aggregated.set_index("policy").loc[present]
    labels = present
    means = grouped["mean"].to_numpy()
    errors = np.array(
        [
            (
                grouped.loc[label, "mean"]
                - grouped.loc[label, "ci_low"],
                grouped.loc[label, "ci_high"]
                - grouped.loc[label, "mean"],
            )
            for label in present
        ]
    ).T
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        labels,
        means,
        yerr=errors,
        capsize=5,
        color="#4C72B0",
    )
    axis.set_ylabel("mean net objective (per episode)")
    axis.set_title(
        "Policy comparison on the event-driven market-making simulator"
    )
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "comparison.png", dpi=140)
    plt.close(fig)
