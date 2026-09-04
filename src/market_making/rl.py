"""Actor-Critic learners for the event-driven market maker.

Algorithm 1 (continuous-time Linear-MC) is the main event-driven learner: a
linear residual Critic is fitted by pooled Monte-Carlo least squares while the
softmax Actor is updated with the score-function advantage evaluated at
realised fills plus the entropy-gradient running term.  Algorithm 2
(discretised-time Actor-Critic) is the fixed-grid benchmark using semi-
gradient TD(0).  Both follow the dissertation's Section 5.2 pseudocode; the
Linear-Pair parametrisation of Section 5.3 is used for the Actor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .intensity_env import (
    MMEnvironmentConfig,
    QuoteItem,
    effective_portfolio,
    elementary_items,
    feasible_portfolios,
    portfolio_mnl_probabilities,
)


def critic_basis(t: float, q: int, config: MMEnvironmentConfig) -> np.ndarray:
    """Six residual-Critic features vanishing at the terminal time.

    Two orthonormal time functions of tau = 1 - t/T (Gram-Schmidt of
    {tau, tau^2} on L^2[0,1]) are crossed with the spatial functions
    {1, q/q_bar, (q/q_bar)^2}, matching Section 5.3.1 / Experiment I.
    """
    tau = 1.0 - float(t) / config.horizon
    tau = min(max(tau, 0.0), 1.0)
    time_one = math.sqrt(3.0) * tau
    time_two = math.sqrt(80.0) * (tau * tau - 0.75 * tau)
    scaled_q = q / config.inventory_cap
    return np.array(
        [
            time_one * 1.0,
            time_one * scaled_q,
            time_one * scaled_q * scaled_q,
            time_two * 1.0,
            time_two * scaled_q,
            time_two * scaled_q * scaled_q,
        ]
    )


class LinearCritic:
    """V_w(t, q) = phi(t, q).T w with a ridge regularised fit."""

    def __init__(self, config: MMEnvironmentConfig) -> None:
        self.config = config
        self.weights = np.zeros(6)

    def predict(self, t: float, q: int) -> float:
        return float(critic_basis(t, q, self.config) @ self.weights)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        ridge: float = 1e-4,
    ) -> None:
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must share rows.")
        design = features.T @ features
        right = features.T @ targets
        penalty = ridge * np.eye(features.shape[1])
        self.weights = np.linalg.solve(design + penalty, right)


class PairwiseActor:
    """Softmax Actor with Linear-Pair logits (Section 5.3.2)."""

    def __init__(
        self,
        config: MMEnvironmentConfig,
        items: tuple[QuoteItem, ...] | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config
        self.items = items or elementary_items(config)
        self.rng = np.random.default_rng(seed)
        count = len(self.items)
        self.pair_count = 1 + count * (count + 1) // 2
        self.parameters = np.zeros((self.pair_count, 2))
        self._feasible = {
            q: feasible_portfolios(q, self.items, config)
            for q in range(-config.inventory_cap, config.inventory_cap + 1)
        }
        self._features = {}

    def feature(self, portfolio: tuple[int, ...]) -> np.ndarray:
        if portfolio in self._features:
            return self._features[portfolio]
        vector = np.zeros(self.pair_count)
        vector[0] = 1.0
        offset = 1
        count = len(self.items)
        for first in range(count):
            for second in range(first, count):
                if first in portfolio and second in portfolio:
                    vector[offset] = 1.0
                offset += 1
        self._features[portfolio] = vector
        return vector

    def _inventory_feature(self, q: int) -> np.ndarray:
        return np.array([1.0, q / self.config.inventory_cap])

    def logits(self, q: int, portfolio: tuple[int, ...]) -> float:
        return float(
            self.feature(portfolio)
            @ self.parameters
            @ self._inventory_feature(q)
        )

    def probabilities(self, q: int) -> dict[tuple[int, ...], float]:
        portfolios = self._feasible[q]
        logits = np.array([self.logits(q, p) for p in portfolios])
        shifted = logits - np.max(logits)
        weights = np.exp(shifted)
        probabilities = weights / weights.sum()
        return dict(zip(portfolios, probabilities.tolist()))

    def log_probability(self, q: int, portfolio: tuple[int, ...]) -> float:
        probabilities = self.probabilities(q)
        if portfolio not in probabilities:
            raise ValueError("portfolio is infeasible at this inventory.")
        return math.log(probabilities[portfolio])

    def sample(self, q: int, rng: np.random.Generator | None = None) -> tuple[int, ...]:
        generator = rng or self.rng
        probabilities = self.probabilities(q)
        keys = list(probabilities)
        values = np.array([probabilities[key] for key in keys])
        return keys[int(generator.choice(len(keys), p=values))]

    def greedy_action(self, q: int) -> tuple[int, ...]:
        probabilities = self.probabilities(q)
        return max(probabilities, key=probabilities.get)

    def act(self, q: int, t: float) -> tuple[int, ...]:
        """Deterministic greedy evaluation action used by the environment."""
        return self.greedy_action(q)

    def entropy(self, q: int) -> float:
        probabilities = list(self.probabilities(q).values())
        return float(-sum(p * math.log(p) for p in probabilities))

    def score_gradient(self, q: int, portfolio: tuple[int, ...]) -> np.ndarray:
        """d log pi(a|q)/dB = (xi(a) - E_pi[xi]) outer zeta(q)."""
        probabilities = self.probabilities(q)
        expected = np.zeros(self.pair_count)
        for candidate, probability in probabilities.items():
            expected += probability * self.feature(candidate)
        difference = self.feature(portfolio) - expected
        return np.outer(difference, self._inventory_feature(q))

    def entropy_gradient(self, q: int) -> np.ndarray:
        """dH(pi(.|q))/dB in closed form over the feasible set."""
        probabilities = self.probabilities(q)
        entropy_value = self.entropy(q)
        gradient = np.zeros_like(self.parameters)
        for candidate, probability in probabilities.items():
            log_prob = math.log(probability)
            weight = probability * (log_prob + 1.0 + entropy_value)
            gradient -= weight * np.outer(
                self.feature(candidate),
                self._inventory_feature(q),
            )
        return gradient


@dataclass
class TimelineEvent:
    time: float
    q_before: int
    q_after: int
    spread: float
    portfolio: tuple[int, ...]
    filled: bool


@dataclass(frozen=True)
class LinearMCTrainConfig:
    entropy_coefficient: float = 5e-3
    actor_learning_rate: float = 3e-3
    ridge: float = 1e-4
    forgetting: float = 0.99
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8


def collect_timeline(
    actor: PairwiseActor,
    config: MMEnvironmentConfig,
    rng: np.random.Generator,
) -> tuple[list[TimelineEvent], float]:
    """Event-driven trajectory with a portfolio sampled at every arrival."""
    items = actor.items
    events = []
    t = 0.0
    q = 0
    while True:
        delay = rng.exponential(1.0 / config.arrival_rate)
        t_next = t + delay
        if t_next >= config.horizon:
            break
        q_before = q
        portfolio = actor.sample(q, rng)
        portfolio = effective_portfolio(portfolio, items, q, config)
        probabilities = portfolio_mnl_probabilities(
            portfolio,
            items,
            config,
        )
        keys = list(probabilities)
        values = np.array([probabilities[key] for key in keys])
        outcome = keys[int(rng.choice(len(keys), p=values))]
        spread = 0.0
        filled = False
        if outcome != -1:
            item = items[outcome]
            spread = item.half_spread
            filled = True
            q += 1 if item.side == "bid" else -1
        events.append(
            TimelineEvent(
                time=float(t_next),
                q_before=q_before,
                q_after=q,
                spread=spread,
                portfolio=portfolio,
                filled=filled,
            )
        )
        t = t_next
    return events, float(t)


def _monte_carlo_targets(
    events: list[TimelineEvent],
    config: MMEnvironmentConfig,
) -> tuple[list[float], list[tuple[float, int]]]:
    """Return-to-go targets for post-fill/arrival and initial states."""
    n = len(events)
    states = [(0.0, 0)] if n == 0 else []
    targets = [0.0] if n == 0 else []
    if n == 0:
        return targets, states
    tail_penalty = (
        config.inventory_penalty
        * events[-1].q_after**2
        * (config.horizon - events[-1].time)
    )
    post_values = [0.0] * n
    post_values[n - 1] = -tail_penalty
    for j in range(n - 2, -1, -1):
        gap = events[j + 1].time - events[j].time
        penalty = (
            config.inventory_penalty
            * events[j].q_after**2
            * gap
        )
        post_values[j] = (
            post_values[j + 1]
            + events[j + 1].spread
            - penalty
        )
    states.append((0.0, 0))
    # Inventory starts at zero, so the first interval carries no penalty.
    targets.append(events[0].spread + post_values[0])
    for j in range(n):
        states.append((events[j].time, events[j].q_after))
        targets.append(post_values[j])
    return targets, states


def _advantage(
    event: TimelineEvent,
    critic: LinearCritic,
) -> float:
    return (
        event.spread
        + critic.predict(event.time, event.q_after)
        - critic.predict(event.time, event.q_before)
    )


def _actor_batch_gradient(
    batch: list[list[TimelineEvent]],
    actor: PairwiseActor,
    critic: LinearCritic,
    config: MMEnvironmentConfig,
    entropy_coefficient: float,
) -> np.ndarray:
    gradient = np.zeros_like(actor.parameters)
    for events in batch:
        t_previous = 0.0
        q_previous = 0
        for event in events:
            gap = event.time - t_previous
            gradient += (
                entropy_coefficient
                * gap
                * actor.entropy_gradient(q_previous)
            )
            if event.filled:
                gradient += _advantage(event, critic) * actor.score_gradient(
                    event.q_before,
                    event.portfolio,
                )
            t_previous = event.time
            q_previous = event.q_after
        tail = config.horizon - t_previous
        if tail > 0.0:
            gradient += (
                entropy_coefficient
                * tail
                * actor.entropy_gradient(q_previous)
            )
    return gradient / max(len(batch), 1)


def _adam_update(
    parameters: np.ndarray,
    gradient: np.ndarray,
    state: dict,
    step: int,
    learning_rate: float,
    config: LinearMCTrainConfig,
) -> None:
    state["m"] = (
        config.adam_beta1 * state["m"]
        + (1.0 - config.adam_beta1) * gradient
    )
    state["v"] = (
        config.adam_beta2 * state["v"]
        + (1.0 - config.adam_beta2) * gradient * gradient
    )
    m_hat = state["m"] / (1.0 - config.adam_beta1**step)
    v_hat = state["v"] / (1.0 - config.adam_beta2**step)
    parameters += (
        learning_rate
        * m_hat
        / (np.sqrt(v_hat) + config.adam_epsilon)
    )


def train_linear_mc(
    config: MMEnvironmentConfig,
    episodes: int,
    batch_size: int = 20,
    seed: int = 0,
    train_config: LinearMCTrainConfig = LinearMCTrainConfig(),
) -> PairwiseActor:
    """Train the event-driven Linear-MC Actor-Critic (Algorithm 1)."""
    if episodes < 1 or batch_size < 1:
        raise ValueError("episodes and batch_size must be positive.")
    rng = np.random.default_rng(seed)
    actor = PairwiseActor(config, seed=seed)
    critic = LinearCritic(config)
    adam_state = {
        "m": np.zeros_like(actor.parameters),
        "v": np.zeros_like(actor.parameters),
    }
    design_sum = np.zeros((6, 6))
    target_sum = np.zeros(6)
    weight_sum = 0.0
    batch: list[list[TimelineEvent]] = []
    update_step = 0
    for episode_index in range(1, episodes + 1):
        events, _ = collect_timeline(actor, config, rng)
        batch.append(events)
        if episode_index % batch_size == 0:
            features_rows = []
            target_rows = []
            for events in batch:
                targets, states = _monte_carlo_targets(events, config)
                features_rows.extend(
                    critic_basis(float(t), int(q), config)
                    for t, q in states
                )
                target_rows.extend(targets)
            features = np.array(features_rows)
            targets = np.array(target_rows)
            design_sum = (
                train_config.forgetting * design_sum
                + features.T @ features
            )
            target_sum = (
                train_config.forgetting * target_sum
                + features.T @ targets
            )
            weight_sum = (
                train_config.forgetting * weight_sum
                + 1.0
            )
            ridge = train_config.ridge * weight_sum
            critic.weights = np.linalg.solve(
                design_sum + ridge * np.eye(6),
                target_sum,
            )
            gradient = _actor_batch_gradient(
                batch,
                actor,
                critic,
                config,
                train_config.entropy_coefficient,
            )
            update_step += 1
            _adam_update(
                actor.parameters,
                gradient,
                adam_state,
                update_step,
                train_config.actor_learning_rate,
                train_config,
            )
            batch = []
    return actor


def train_discrete_ac(
    config: MMEnvironmentConfig,
    episodes: int,
    batch_size: int = 20,
    dt: float = 1.0,
    seed: int = 0,
    entropy_coefficient: float = 5e-3,
    critic_learning_rate: float = 1e-2,
    actor_learning_rate: float = 1e-2,
) -> PairwiseActor:
    """Train the discretised-time Actor-Critic benchmark (Algorithm 2)."""
    if episodes < 1 or batch_size < 1:
        raise ValueError("episodes and batch_size must be positive.")
    rng = np.random.default_rng(seed)
    actor = PairwiseActor(config, seed=seed + 1)
    critic = LinearCritic(config)
    actor_state = {
        "m": np.zeros_like(actor.parameters),
        "v": np.zeros_like(actor.parameters),
    }
    grid_steps = max(1, int(math.ceil(config.horizon / dt)))
    update_step = 0

    def run_grid_episode_trajectory() -> list[dict]:
        rows = []
        t = 0.0
        q = 0
        for n in range(grid_steps):
            cell_end = min((n + 1) * dt, config.horizon)
            if cell_end <= t:
                break
            q_decision = q
            action = actor.sample(q_decision, rng)
            spread_cell = 0.0
            penalty_cell = 0.0
            while True:
                delay = rng.exponential(1.0 / config.arrival_rate)
                t_next = t + delay
                if t_next >= cell_end:
                    penalty_cell += (
                        config.inventory_penalty
                        * q
                        * q
                        * (cell_end - t)
                    )
                    t = cell_end
                    break
                penalty_cell += (
                    config.inventory_penalty * q * q * (t_next - t)
                )
                portfolio = effective_portfolio(
                    action,
                    actor.items,
                    q,
                    config,
                )
                probabilities = portfolio_mnl_probabilities(
                    portfolio,
                    actor.items,
                    config,
                )
                keys = list(probabilities)
                values = np.array([probabilities[key] for key in keys])
                outcome = keys[int(rng.choice(len(keys), p=values))]
                if outcome != -1:
                    item = actor.items[outcome]
                    spread_cell += item.half_spread
                    q += 1 if item.side == "bid" else -1
                t = t_next
            rows.append(
                {
                    "t": n * dt,
                    "q": q_decision,
                    "q_after": q,
                    "action": action,
                    "reward": spread_cell - penalty_cell,
                }
            )
        return rows

    batch_rows = []
    for episode_index in range(1, episodes + 1):
        batch_rows.extend(run_grid_episode_trajectory())
        if episode_index % batch_size == 0:
            critic_gradient = np.zeros(6)
            actor_gradient = np.zeros_like(actor.parameters)
            for row in batch_rows:
                t = row["t"]
                q = row["q"]
                q_after = row["q_after"]
                t_next = min(t + dt, config.horizon)
                features_current = critic_basis(t, q, config)
                features_next = critic_basis(t_next, q_after, config)
                value_current = float(features_current @ critic.weights)
                value_next = float(features_next @ critic.weights)
                advantage = row["reward"] + value_next - value_current
                entropy_bonus = (
                    entropy_coefficient
                    * dt
                    * actor.entropy_gradient(q)
                )
                critic_gradient += (
                    advantage * features_current
                )
                actor_gradient += (
                    advantage * actor.score_gradient(q, row["action"])
                    + entropy_bonus
                )
            critic.weights += (
                critic_learning_rate
                * critic_gradient
                / max(len(batch_rows), 1)
            )
            update_step += 1
            _adam_update(
                actor.parameters,
                actor_gradient / max(len(batch_rows), 1),
                actor_state,
                update_step,
                actor_learning_rate,
                LinearMCTrainConfig(),
            )
            batch_rows = []
    return actor
