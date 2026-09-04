"""Dynamic-programming benchmark for the single-asset market maker.

The residual HJB of the dissertation reduces to a finite-state ODE system
(equation 6.1).  This module solves it backward in time with an explicit Euler
scheme on the inventory grid, reproduces the closed-form exponential-fill
cross-check (equation 6.4), and provides the short-sighted greedy action map.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import expm

from .intensity_env import (
    MMEnvironmentConfig,
    QuoteItem,
    elementary_items,
    feasible_portfolios,
    portfolio_mnl_probabilities,
)


@dataclass(frozen=True)
class DPSolution:
    values: np.ndarray  # (n_steps + 1, 2*q_bar + 1), row 0 is time 0
    action_map: np.ndarray  # object array with the arg-max portfolio tuples
    n_steps: int
    terminal_zero: bool


def _state_count(config: MMEnvironmentConfig) -> int:
    return 2 * config.inventory_cap + 1


def _q_index(q: int, config: MMEnvironmentConfig) -> int:
    return q + config.inventory_cap


def mnl_dp_solver(
    config: MMEnvironmentConfig,
    items: tuple[QuoteItem, ...] | None = None,
    n_steps: int = 10_000,
) -> DPSolution:
    """Solve the finite-state residual ODE system (6.1) backward in time."""
    if items is None:
        items = elementary_items(config)
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    cap = config.inventory_cap
    n_q = _state_count(config)
    dt = config.horizon / n_steps

    # Precompute, per inventory state, every feasible portfolio's jump terms:
    # (target inventory indexes, effective fill intensities lambda * P_j).
    transitions = {}
    spreads = {}
    total_intensities = {}
    for q in range(-cap, cap + 1):
        portfolio_terms = []
        portfolio_spreads = []
        portfolio_totals = []
        for portfolio in feasible_portfolios(q, items, config):
            probabilities = portfolio_mnl_probabilities(
                portfolio,
                items,
                config,
            )
            terms = []
            spread_sum = 0.0
            total_intensity = 0.0
            for item_index, probability in probabilities.items():
                if item_index == -1:
                    continue
                intensity = config.arrival_rate * probability
                total_intensity += intensity
                item = items[item_index]
                spread_sum += intensity * item.half_spread
                impact = 1 if item.side == "bid" else -1
                terms.append(
                    (q + impact, intensity)
                )
            portfolio_terms.append(tuple(terms))
            portfolio_spreads.append(spread_sum)
            portfolio_totals.append(total_intensity)
        transitions[q] = portfolio_terms
        spreads[q] = portfolio_spreads
        total_intensities[q] = portfolio_totals

    values = np.zeros((n_steps + 1, n_q))
    action_map = np.empty((n_steps + 1, n_q), dtype=object)
    action_map[-1, :] = [() for _ in range(n_q)]
    for step in range(n_steps - 1, -1, -1):
        next_row = values[step + 1]
        current = values[step]
        action_row = action_map[step]
        for q in range(-cap, cap + 1):
            q_index = _q_index(q, config)
            best_value = -math.inf
            best_action = ()
            for portfolio, terms, spread_sum, total_intensity in zip(
                feasible_portfolios(q, items, config),
                transitions[q],
                spreads[q],
                total_intensities[q],
            ):
                jump = spread_sum + sum(
                    intensity * next_row[_q_index(target, config)]
                    for target, intensity in terms
                ) - total_intensity * next_row[q_index]
                # Compare against continuing at q (jump term excludes the
                # no-change outside option by construction).
                candidate = jump
                if candidate > best_value:
                    best_value = candidate
                    best_action = portfolio
            current[q_index] = (
                next_row[q_index]
                + dt
                * (
                    best_value
                    - config.inventory_penalty * q * q
                )
            )
            action_row[q_index] = best_action
    return DPSolution(
        values=values,
        action_map=action_map,
        n_steps=n_steps,
        terminal_zero=bool(np.allclose(values[-1], 0.0)),
    )


def _exponential_ode_step(
    phi: float,
    kappa: float,
    activity: float,
    q: int,
    cap: int,
    next_values: np.ndarray,
) -> float:
    """RHS term of the reduced ODE (6.2) at inventory q."""
    term = 0.0
    base = activity * math.exp(-1.0) / kappa
    for target in (q + 1, q - 1):
        if -cap <= target <= cap:
            term += math.exp(
                kappa
                * (
                    next_values[target + cap]
                    - next_values[q + cap]
                )
            )
    return phi * q * q - base * term


def exponential_fill_dp_value(
    phi: float,
    kappa: float,
    activity: float,
    inventory_cap: int,
    horizon: float,
    n_steps: int = 10_000,
) -> np.ndarray:
    """Euler solution of equation (6.2) on the inventory grid."""
    n_q = 2 * inventory_cap + 1
    values = np.zeros((n_steps + 1, n_q))
    dt = horizon / n_steps
    for step in range(n_steps - 1, -1, -1):
        next_row = values[step + 1]
        current = values[step]
        for q in range(-inventory_cap, inventory_cap + 1):
            q_index = q + inventory_cap
            derivative = _exponential_ode_step(
                phi=phi,
                kappa=kappa,
                activity=activity,
                q=q,
                cap=inventory_cap,
                next_values=next_row,
            )
            current[q_index] = (
                next_row[q_index] - dt * derivative
            )
    return values


def exponential_fill_closed_form_value(
    phi: float,
    kappa: float,
    activity: float,
    inventory_cap: int,
    horizon: float,
) -> np.ndarray:
    """Closed-form solution (6.4) via the matrix exponential."""
    n_q = 2 * inventory_cap + 1
    matrix = np.zeros((n_q, n_q))
    for q in range(-inventory_cap, inventory_cap + 1):
        index = q + inventory_cap
        matrix[index, index] = kappa * phi * q * q
        off = -activity * math.exp(-1.0)
        for target in (q + 1, q - 1):
            if -inventory_cap <= target <= inventory_cap:
                matrix[index, target + inventory_cap] = off
    omega = expm(-matrix * horizon) @ np.ones(n_q)
    return (1.0 / kappa) * np.log(omega)


def greedy_action_map(
    config: MMEnvironmentConfig,
    items: tuple[QuoteItem, ...] | None = None,
) -> np.ndarray:
    """Model-based short-sighted action map: maximise expected immediate spread."""
    if items is None:
        items = elementary_items(config)
    cap = config.inventory_cap
    n_q = 2 * cap + 1
    action_map = np.empty((1, n_q), dtype=object)
    for q in range(-cap, cap + 1):
        q_index = q + cap
        best_value = -math.inf
        best_action = ()
        for portfolio in feasible_portfolios(q, items, config):
            probabilities = portfolio_mnl_probabilities(
                portfolio,
                items,
                config,
            )
            expected = sum(
                config.arrival_rate
                * probability
                * items[item_index].half_spread
                for item_index, probability in probabilities.items()
                if item_index != -1
            )
            if expected > best_value:
                best_value = expected
                best_action = portfolio
        action_map[0, q_index] = best_action
    return action_map


def dp_value_at_zero(solver: DPSolution, config: MMEnvironmentConfig) -> float:
    """V*(0, 0) = residual value starting from zero inventory."""
    return float(solver.values[0, _q_index(0, config)])
