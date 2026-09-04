"""Run the event-driven RL vs DP policy study and write CSV/JSON outputs.

Uses the dissertation's Experiment I environment parameters
(lambda=0.9, kappa=3, spreads {0.3, 0.6}, q_bar=5, phi=5e-3, T=15) so the
numerically converged DP benchmark can be cross-checked against the reported
V*(0,0) ~ 1.90299.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.market_making.intensity_env import MMEnvironmentConfig
from src.market_making.study import run_policy_comparison, save_study_results


def main() -> None:
    config = MMEnvironmentConfig(
        arrival_rate=0.9,
        kappa=3.0,
        outside_weight=1.0,
        half_spreads=(0.3, 0.6),
        inventory_cap=5,
        horizon=15.0,
        inventory_penalty=5e-3,
    )
    result = run_policy_comparison(
        config=config,
        seeds=(123, 456, 789),
        n_eval_episodes=300,
        dp_steps=10_000,
        rl_episodes=5_000,
        rl_batch_size=20,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = save_study_results(
        result,
        Path("outputs") / f"market_making_study_{stamp}",
    )
    print(
        f"DP benchmark V*(0,0): {result['dp_value_zero']:.5f} "
        "(dissertation reference 1.90299)"
    )
    print(result["aggregated"].to_string(index=False))
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
