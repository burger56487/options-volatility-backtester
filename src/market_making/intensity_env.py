"""Event-driven single-asset market-making environment.

Implements the dissertation's Section 5.1 simulation framework: liquidity
takers arrive as a Poisson process, are routed across the posted quote
portfolio by the multinomial-logit model, and only fills change inventory and
captured spread.  No-fill arrivals still advance time and accrue the running
inventory penalty, matching the continuous-time objective.  A fixed-grid
variant is included for the discretised-time Actor-Critic benchmark
(Algorithm 2 of Section 4.2).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class QuoteItem:
    side: str  # "bid" or "ask"
    half_spread: float


@dataclass(frozen=True)
class MMEnvironmentConfig:
    arrival_rate: float = 0.9
    kappa: float = 3.0
    outside_weight: float = 1.0
    half_spreads: tuple[float, ...] = (0.3, 0.6)
    inventory_cap: int = 5
    horizon: float = 15.0
    inventory_penalty: float = 5e-3

    def __post_init__(self) -> None:
        if self.arrival_rate <= 0.0:
            raise ValueError("arrival_rate must be positive.")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive.")
        if self.outside_weight < 0.0:
            raise ValueError("outside_weight must be non-negative.")
        if not self.half_spreads:
            raise ValueError("half_spreads must be non-empty.")
        if any(spread <= 0 for spread in self.half_spreads):
            raise ValueError("half_spreads must be positive.")
        if self.inventory_cap < 1:
            raise ValueError("inventory_cap must be at least 1.")
        if self.horizon <= 0.0:
            raise ValueError("horizon must be positive.")
        if self.inventory_penalty < 0.0:
            raise ValueError("inventory_penalty must be non-negative.")


@dataclass(frozen=True)
class FillEvent:
    time: float
    q_before: int
    q_after: int
    spread: float
    item_index: int


@dataclass(frozen=True)
class EpisodeResult:
    net_objective: float
    spread_total: float
    penalty_total: float
    fill_count: int
    n_arrivals: int
    final_inventory: int
    terminal_time: float
    fills: tuple[FillEvent, ...]
    grid_steps: int | None = None


def elementary_items(config: MMEnvironmentConfig) -> tuple[QuoteItem, ...]:
    """Bid items followed by ask items, tightest spread first per side."""
    items = []
    for side in ("bid", "ask"):
        for spread in config.half_spreads:
            items.append(QuoteItem(side=side, half_spread=float(spread)))
    return tuple(items)


def feasible_portfolios(
    q: int,
    items: tuple[QuoteItem, ...],
    config: MMEnvironmentConfig,
) -> tuple[tuple[int, ...], ...]:
    """All feasible quote portfolios at inventory q.

    A portfolio contains at most one bid and one ask item; items whose fill
    would push inventory outside [-q_bar, q_bar] are excluded.
    """
    half = len(items) // 2
    bid_choices = [
        index
        for index, item in enumerate(items[:half])
        if item.side == "bid" and q + 1 <= config.inventory_cap
    ]
    ask_choices = [
        index
        for index, item in enumerate(items[half:], start=half)
        if item.side == "ask" and q - 1 >= -config.inventory_cap
    ]
    portfolios = []
    for bid in (None, *bid_choices):
        for ask in (None, *ask_choices):
            portfolio = tuple(
                sorted(index for index in (bid, ask) if index is not None)
            )
            portfolios.append(portfolio)
    return tuple(portfolios)


def effective_portfolio(
    portfolio: tuple[int, ...],
    items: tuple[QuoteItem, ...],
    q: int,
    config: MMEnvironmentConfig,
) -> tuple[int, ...]:
    """Drop quote items that are infeasible at the current inventory."""
    kept = []
    for index in portfolio:
        item = items[index]
        if item.side == "bid" and q + 1 <= config.inventory_cap:
            kept.append(index)
        elif item.side == "ask" and q - 1 >= -config.inventory_cap:
            kept.append(index)
    return tuple(sorted(kept))


def portfolio_mnl_probabilities(
    portfolio: tuple[int, ...],
    items: tuple[QuoteItem, ...],
    config: MMEnvironmentConfig,
) -> dict[int, float]:
    """Multinomial-logit routing probabilities with the outside option."""
    total = config.outside_weight
    weights = {}
    for index in portfolio:
        weight = math.exp(-config.kappa * items[index].half_spread)
        weights[index] = weight
        total += weight
    output = {index: weight / total for index, weight in weights.items()}
    output[-1] = config.outside_weight / total
    return output


def _sample_outcome(probabilities: dict[int, float], rng: np.random.Generator) -> int:
    keys = list(probabilities)
    values = np.array([probabilities[key] for key in keys])
    return int(keys[rng.choice(len(keys), p=values)])


def run_episode(policy, config: MMEnvironmentConfig, rng: np.random.Generator) -> EpisodeResult:
    """Run one event-driven episode; decisions are made at each arrival."""
    items = elementary_items(config)
    t = 0.0
    q = 0
    n_arrivals = 0
    spread_total = 0.0
    penalty_total = 0.0
    fills = []
    while True:
        delay = rng.exponential(1.0 / config.arrival_rate)
        t_next = t + delay
        if t_next >= config.horizon:
            break
        penalty_total += (
            config.inventory_penalty * q * q * (t_next - t)
        )
        q_before = q
        portfolio = policy.act(q, t_next)
        portfolio = effective_portfolio(portfolio, items, q, config)
        probabilities = portfolio_mnl_probabilities(
            portfolio,
            items,
            config,
        )
        outcome = _sample_outcome(probabilities, rng)
        if outcome != -1:
            item = items[outcome]
            spread_total += item.half_spread
            q += 1 if item.side == "bid" else -1
            fills.append(
                FillEvent(
                    time=float(t_next),
                    q_before=q_before,
                    q_after=q,
                    spread=item.half_spread,
                    item_index=outcome,
                )
            )
        n_arrivals += 1
        t = t_next
    penalty_total += (
        config.inventory_penalty * q * q * (config.horizon - t)
    )
    return EpisodeResult(
        net_objective=float(spread_total - penalty_total),
        spread_total=float(spread_total),
        penalty_total=float(penalty_total),
        fill_count=len(fills),
        n_arrivals=n_arrivals,
        final_inventory=q,
        terminal_time=float(t),
        fills=tuple(fills),
    )


def run_grid_episode(
    policy,
    config: MMEnvironmentConfig,
    dt: float,
    rng: np.random.Generator,
) -> EpisodeResult:
    """Run one episode on the fixed grid used by Algorithm 2.

    Quotes are sampled at grid times and frozen over the cell while Poisson
    arrivals execute against them through the same MNL routing.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    items = elementary_items(config)
    steps = max(1, int(math.ceil(config.horizon / dt)))
    t = 0.0
    q = 0
    n_arrivals = 0
    spread_total = 0.0
    penalty_total = 0.0
    fills = []
    processed = 0
    for n in range(steps):
        cell_end = min((n + 1) * dt, config.horizon)
        if cell_end <= t:
            break
        portfolio = policy.act(q, t)
        while True:
            delay = rng.exponential(1.0 / config.arrival_rate)
            t_next = t + delay
            if t_next >= cell_end:
                penalty_total += (
                    config.inventory_penalty
                    * q
                    * q
                    * (cell_end - t)
                )
                t = cell_end
                break
            penalty_total += (
                config.inventory_penalty * q * q * (t_next - t)
            )
            q_before = q
            current = effective_portfolio(portfolio, items, q, config)
            probabilities = portfolio_mnl_probabilities(
                current,
                items,
                config,
            )
            outcome = _sample_outcome(probabilities, rng)
            if outcome != -1:
                item = items[outcome]
                spread_total += item.half_spread
                q += 1 if item.side == "bid" else -1
                fills.append(
                    FillEvent(
                        time=float(t_next),
                        q_before=q_before,
                        q_after=q,
                        spread=item.half_spread,
                        item_index=outcome,
                    )
                )
            n_arrivals += 1
            t = t_next
        processed += 1
    if t < config.horizon:
        penalty_total += (
            config.inventory_penalty * q * q * (config.horizon - t)
        )
    return EpisodeResult(
        net_objective=float(spread_total - penalty_total),
        spread_total=float(spread_total),
        penalty_total=float(penalty_total),
        fill_count=len(fills),
        n_arrivals=n_arrivals,
        final_inventory=q,
        terminal_time=float(min(t, config.horizon)),
        fills=tuple(fills),
        grid_steps=processed,
    )
