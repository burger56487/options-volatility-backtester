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
