"""Moving-block bootstrap confidence intervals with block-size heuristic."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def block_size_heuristic(n: int) -> int:
    return max(1, int(math.ceil(n ** (1.0 / 3.0))))


def moving_block_samples(
    returns,
    n_samples: int = 2_000,
    block_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    clean = np.asarray(returns.dropna(), dtype=float)
    n = clean.size
    if n == 0:
        raise ValueError("No observations for bootstrap.")
    if block_size is None:
        block_size = block_size_heuristic(n)
    block_size = min(max(block_size, 1), n)
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_size))
    starts = rng.integers(
        0,
        n - block_size + 1,
        size=(n_samples, n_blocks),
    )
    out = np.empty((n_samples, n))
    for i in range(n_samples):
        parts = [
            clean[s : s + block_size] for s in starts[i]
        ]
        out[i] = np.concatenate(parts)[:n]
    return out


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
