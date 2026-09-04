"""VaR / ES measures with sample guards and a unified result object."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


MINIMUM_OBSERVATIONS = 20


@dataclass(frozen=True)
class RiskMeasureResult:
    method: str
    var: float
    expected_shortfall: float
    confidence_level: float
    n_observations: int
    insufficient_sample: bool = False
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "var": self.var,
            "expected_shortfall": self.expected_shortfall,
            "confidence_level": self.confidence_level,
            "n_observations": self.n_observations,
            "insufficient_sample": self.insufficient_sample,
            "params": self.params,
        }


def delta_normal_var(
    exposure: float,
    daily_volatility: float,
    confidence_level: float = 0.95,
) -> RiskMeasureResult:
    """Delta-normal VaR for one risk factor."""
    from scipy.stats import norm

    z = float(-norm.ppf(1.0 - confidence_level))
    var = abs(exposure) * daily_volatility * z
    return RiskMeasureResult(
        method="delta_normal",
        var=var,
        expected_shortfall=var,
        confidence_level=confidence_level,
        n_observations=0,
        params={"exposure": exposure, "daily_volatility": daily_volatility},
    )


def delta_gamma_var(
    exposure: float,
    gamma: float,
    spot: float,
    daily_volatility: float,
    confidence_level: float = 0.95,
) -> RiskMeasureResult:
    """Delta-gamma VaR via quadratic approximation (Cornish-Fisher style)."""
    z = 1.645 if confidence_level == 0.95 else 2.326
    delta_term = abs(exposure) * daily_volatility * z
    gamma_term = 0.5 * gamma * (daily_volatility * spot * z) ** 2
    var = delta_term - gamma_term  # long gamma reduces VaR
    return RiskMeasureResult(
        method="delta_gamma",
        var=max(var, 0.0),
        expected_shortfall=max(var, 0.0),
        confidence_level=confidence_level,
        n_observations=0,
        params={
            "exposure": exposure,
            "gamma": gamma,
            "spot": spot,
            "daily_volatility": daily_volatility,
        },
    )


def historical_var(
    pnl: pd.Series,
    confidence_level: float = 0.95,
) -> RiskMeasureResult:
    """Historical-simulation VaR/ES with a minimum-observation guard."""
    values = pnl.dropna().to_numpy(dtype=float)
    n = int(values.size)
    if n < MINIMUM_OBSERVATIONS:
        return RiskMeasureResult(
            method="historical",
            var=float("nan"),
            expected_shortfall=float("nan"),
            confidence_level=confidence_level,
            n_observations=n,
            insufficient_sample=True,
        )
    quantile = np.quantile(values, 1.0 - confidence_level)
    tail = values[values <= quantile]
    return RiskMeasureResult(
        method="historical",
        var=float(max(0.0, -quantile)),
        expected_shortfall=float(max(0.0, -tail.mean()))
        if tail.size
        else 0.0,
        confidence_level=confidence_level,
        n_observations=n,
    )


def filtered_historical_var(
    pnl: pd.Series,
    confidence_level: float = 0.95,
    decay: float = 0.97,
) -> RiskMeasureResult:
    """EWMA-weighted historical simulation (recent losses weighted more)."""
    values = pnl.dropna().to_numpy(dtype=float)
    n = int(values.size)
    if n < MINIMUM_OBSERVATIONS:
        return RiskMeasureResult(
            method="filtered_historical",
            var=float("nan"),
            expected_shortfall=float("nan"),
            confidence_level=confidence_level,
            n_observations=n,
            insufficient_sample=True,
        )
    weights = np.array(
        [(1.0 - decay) * decay ** (n - 1 - i) for i in range(n)]
    )
    weights /= weights.sum()
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    idx = int(np.searchsorted(cumulative, 1.0 - confidence_level))
    var = max(0.0, -float(values[order[idx]]))
    tail_mask = order[: idx + 1]
    es = float(
        max(0.0, -np.sum(values[tail_mask] * weights[tail_mask]) / weights[tail_mask].sum())
    )
    return RiskMeasureResult(
        method="filtered_historical",
        var=var,
        expected_shortfall=es,
        confidence_level=confidence_level,
        n_observations=n,
        params={"decay": decay},
    )


def monte_carlo_var(
    pnl_simulator,
    n_scenarios: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> RiskMeasureResult:
    """Monte Carlo VaR via a supplied full-revaluation simulator."""
    if n_scenarios < MINIMUM_OBSERVATIONS:
        return RiskMeasureResult(
            method="monte_carlo",
            var=float("nan"),
            expected_shortfall=float("nan"),
            confidence_level=confidence_level,
            n_observations=n_scenarios,
            insufficient_sample=True,
        )
    scenarios = np.asarray(pnl_simulator(n_scenarios, seed), dtype=float)
    quantile = np.quantile(scenarios, 1.0 - confidence_level)
    tail = scenarios[scenarios <= quantile]
    return RiskMeasureResult(
        method="monte_carlo",
        var=float(max(0.0, -quantile)),
        expected_shortfall=float(max(0.0, -tail.mean()))
        if tail.size
        else 0.0,
        confidence_level=confidence_level,
        n_observations=int(n_scenarios),
    )


def liquidity_adjusted_var(
    market_var: float,
    position_notional: float,
    relative_half_spread: float,
    impact_coefficient: float = 0.0,
    liquidation_fraction: float = 1.0,
) -> float:
    """Add a research-grade liquidation cost to a market-risk VaR.

    The adjustment is ``notional * (half_spread + impact * fraction)``,
    i.e. the cost of exiting ``liquidation_fraction`` of the position into a
    market with a relative half-spread and a linear market-impact term. This
    is a deliberately simplified model (no depth curve, no price resilience)
    and is documented as such in the README.
    """
    if market_var < 0.0:
        raise ValueError("market_var must be non-negative.")
    if position_notional < 0.0:
        raise ValueError("position_notional must be non-negative.")
    if relative_half_spread < 0.0:
        raise ValueError("relative_half_spread must be non-negative.")
    if impact_coefficient < 0.0:
        raise ValueError("impact_coefficient must be non-negative.")
    if not 0.0 <= liquidation_fraction <= 1.0:
        raise ValueError("liquidation_fraction must lie in [0, 1].")
    liquidation_cost = position_notional * (
        relative_half_spread
        + impact_coefficient * liquidation_fraction
    )
    return market_var + liquidation_cost
