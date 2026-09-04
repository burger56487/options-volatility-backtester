"""Moving-block bootstrap confidence intervals with block-size heuristic.

The block sampler and the block-size heuristic live in :mod:`src.statistics`;
this module keeps the performance-oriented wrappers (percentile CI and
annualized Sharpe CI) so both public APIs stay unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.statistics import (
    block_size_heuristic,
    moving_block_bootstrap_samples,
)


def _clean_series(returns) -> pd.Series:
    clean = returns.dropna()
    if len(clean) == 0:
        raise ValueError("No observations for bootstrap.")
    return clean


def moving_block_samples(
    returns,
    n_samples: int = 2_000,
    block_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    clean = _clean_series(returns)
    n = int(clean.size)
    if block_size is None:
        block_size = block_size_heuristic(n)
    return moving_block_bootstrap_samples(
        clean,
        block_size=block_size,
        n_samples=n_samples,
        seed=seed,
    )


def percentile_ci(values: np.ndarray, confidence_level: float):
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.percentile(values, 100 * alpha)),
        float(np.percentile(values, 100 * (1 - alpha))),
    )


def sharpe_ci(
    returns,
    confidence_level: float = 0.95,
    periods_per_year: float = 252.0,
    n_samples: int = 2_000,
    seed: int | None = None,
    risk_free_rate: float = 0.0,
) -> dict:
    clean = returns.dropna()
    n = int(clean.size)
    block_size = min(max(block_size_heuristic(n), 1), n)
    samples = moving_block_samples(
        clean,
        n_samples=n_samples,
        block_size=block_size,
        seed=seed,
    )
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (
            (means - risk_free_rate / periods_per_year)
            / stds
            * math.sqrt(periods_per_year)
        )
    low, high = percentile_ci(sharpe, confidence_level)
    return {
        "annualized_sharpe_ci_low": low,
        "annualized_sharpe_ci_high": high,
        "annualized_sharpe_point": float(
            np.nanmedian(sharpe)
        ),
        "block_size": int(block_size),
    }
