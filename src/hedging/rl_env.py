"""Lightweight gym-style hedge environment and linear policy training."""

from __future__ import annotations

import numpy as np


class HedgeEnv:
    """Discrete-time hedge environment over precomputed delta/spot paths.

    State: [normalised delta, normalised position, volatility state].
    Action: target delta exposure in [-1, 1] (fraction of |delta|).
    Reward: -(hedge PnL residual)^2 - lambda * transaction cost.
    """

    def __init__(
        self,
        spot_path,
        option_delta_path,
        cost_bps: float = 1.0,
        risk_aversion: float = 1.0,
    ) -> None:
        self.spot = np.asarray(spot_path, dtype=float)
        self.delta = np.asarray(option_delta_path, dtype=float)
        self.cost_rate = cost_bps / 10_000.0
        self.risk_aversion = risk_aversion
        self.t = 0
        self.position = 0.0
        self.total_reward = 0.0

    def reset(self) -> np.ndarray:
        self.t = 0
        self.position = 0.0
        self.total_reward = 0.0
        return self._state()

    def _state(self) -> np.ndarray:
        scale = max(abs(self.delta[self.t]), 1.0)
        history = self.delta[max(self.t - 5, 0) : self.t + 1]
        vol_state = (
            float(np.std(np.diff(history)))
            if len(history) >= 2
            else 0.0
        )
        return np.array(
            [
                self.delta[self.t] / scale,
                self.position / scale,
                min(vol_state / 0.05, 1.0),
            ]
        )

    def step(self, action: float):
        if self.t >= len(self.spot) - 1:
            raise RuntimeError("Episode already finished.")
        target = -action * abs(self.delta[self.t])
        trade = target - self.position
        cost = abs(trade) * self.spot[self.t] * self.cost_rate
        d_spot = self.spot[self.t + 1] - self.spot[self.t]
        # Move to next period, then evaluate residual against the new delta.
        hedge_pnl = -self.position * d_spot
        self.position = target
        self.t += 1
        expected_move = -self.delta[self.t - 1] * d_spot
        residual = hedge_pnl - expected_move
        reward = -residual**2 - self.risk_aversion * cost
        self.total_reward += reward
        done = self.t >= len(self.spot) - 1
        return self._state(), float(reward), bool(done), {}


def train_linear_policy(
    env: HedgeEnv,
    episodes: int = 200,
    learning_rate: float = 0.02,
    seed: int = 0,
) -> float:
    """Train a scalar weight mapping normalised delta to target exposure."""
    rng = np.random.default_rng(seed)
    weight = 1.0
    best_weight = weight
    best_reward = -1e18
    for _ in range(episodes):
        env.reset()
        total = 0.0
        while True:
            state = env._state()
            action = np.clip(
                weight * state[0] + rng.normal(0, 0.05),
                -1.0,
                1.0,
            )
            _, reward, done, _ = env.step(float(action))
            total += reward
            if done:
                break
        if total > best_reward:
            best_reward = total
            best_weight = weight
        weight += learning_rate * (total - best_reward) * (1.0 / (1.0 + abs(weight)))
    env.reset()
    return best_weight
