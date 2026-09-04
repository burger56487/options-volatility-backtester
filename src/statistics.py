"""Statistical validation helpers for trade-level backtest results.

The moving-block bootstrap resamples contiguous blocks of trade returns so
that short-range autocorrelation is preserved. A naive i.i.d. bootstrap would
ignore serial dependence and produce over-narrow confidence intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def moving_block_bootstrap_samples(
    values: pd.Series,
    block_size: int = 5,
    n_samples: int = 2_000,
    seed: int | None = None,
) -> np.ndarray:
    """Resample a series with the moving-block bootstrap."""
    clean = values.dropna().to_numpy(dtype=float)
    n = clean.size
    if n == 0:
        raise ValueError("values must not be empty after dropna.")

    block_size = int(min(max(block_size, 1), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size + 1
    starts = rng.integers(0, max_start, size=(n_samples, n_blocks))

    samples = np.empty((n_samples, n))
    for i in range(n_samples):
        chunks = [
            clean[start : start + block_size] for start in starts[i]
        ]
        combined = np.concatenate(chunks)
        samples[i, :] = combined[:n]
    return samples


def block_bootstrap_intervals(
    trade_returns: pd.Series,
    block_size: int = 5,
    n_samples: int = 2_000,
    seed: int | None = None,
    confidence_level: float = 0.95,
    risk_free_rate: float = 0.0,
    trades_per_year: float = 1.0,
) -> dict[str, float]:
    """Return percentile confidence intervals for trade-level metrics.

    The repository-level ``sharpe_like_ratio`` is defined as
    ``(mean_trade_return - risk_free_rate) / trade_return_volatility *
    sqrt(n_trades)`` and its bootstrap interval is computed for that same
    statistic. The annualized estimate scales the un-annualized trade-level
    ratio by ``sqrt(trades_per_year)`` under an explicit trades-per-year
    assumption, so results are never presented as an opaque "Sharpe".
    """
    clean = trade_returns.dropna()
    n = clean.size

    if n < 2:
        return {
            "mean_trade_return_ci_low": float("nan"),
            "mean_trade_return_ci_high": float("nan"),
            "sharpe_like_ratio_ci_low": float("nan"),
            "sharpe_like_ratio_ci_high": float("nan"),
            "annualized_sharpe_estimate": float("nan"),
            "annualized_sharpe_ci_low": float("nan"),
            "annualized_sharpe_ci_high": float("nan"),
            "bootstrap_block_size": float(min(block_size, n)),
            "bootstrap_n_samples": float(n_samples),
            "insufficient_sample": True,
        }

    samples = moving_block_bootstrap_samples(
        values=clean,
        block_size=block_size,
        n_samples=n_samples,
        seed=seed,
    )

    sample_means = samples.mean(axis=1)
    sample_stds = samples.std(axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        trade_sharpe = (sample_means - risk_free_rate) / sample_stds
        sharpe_like = trade_sharpe * np.sqrt(n)
        annualized_sharpe = trade_sharpe * np.sqrt(trades_per_year)

    alpha = (1.0 - confidence_level) / 2.0
    lower = 100.0 * alpha
    upper = 100.0 * (1.0 - alpha)

    def _ci(series: np.ndarray) -> tuple[float, float]:
        finite = series[np.isfinite(series)]
        if finite.size == 0:
            return 0.0, 0.0
        lo, hi = np.percentile(finite, [lower, upper])
        return float(lo), float(hi)

    mean_lo, mean_hi = _ci(sample_means)
    sharpe_lo, sharpe_hi = _ci(sharpe_like)
    annual_lo, annual_hi = _ci(annualized_sharpe)

    return {
        "mean_trade_return_ci_low": mean_lo,
        "mean_trade_return_ci_high": mean_hi,
        "sharpe_like_ratio_ci_low": sharpe_lo,
        "sharpe_like_ratio_ci_high": sharpe_hi,
        "annualized_sharpe_estimate": float(
            np.nanmedian(annualized_sharpe)
            if np.isfinite(annualized_sharpe).any()
            else 0.0
        ),
        "annualized_sharpe_ci_low": annual_lo,
        "annualized_sharpe_ci_high": annual_hi,
        "bootstrap_block_size": float(
            min(block_size, n)
        ),
        "bootstrap_n_samples": float(n_samples),
        "insufficient_sample": False,
    }
