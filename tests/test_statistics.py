import math

import numpy as np
import pandas as pd

from src.statistics import (
    block_bootstrap_intervals,
    moving_block_bootstrap_samples,
)


def test_bootstrap_sample_shape():
    series = pd.Series(np.arange(1.0, 21.0))
    samples = moving_block_bootstrap_samples(
        series,
        block_size=4,
        n_samples=100,
        seed=1,
    )
    assert samples.shape == (100, 20)


def test_bootstrap_mean_is_unbiased():
    series = pd.Series(np.arange(1.0, 21.0))
    samples = moving_block_bootstrap_samples(
        series,
        block_size=3,
        n_samples=5_000,
        seed=42,
    )
    assert abs(float(samples.mean()) - float(series.mean())) < 0.5


def test_intervals_contain_observed_mean():
    rng = np.random.default_rng(7)
    series = pd.Series(rng.normal(0.01, 0.05, size=60))
    intervals = block_bootstrap_intervals(
        series,
        block_size=5,
        n_samples=1_000,
        seed=3,
        confidence_level=0.95,
        risk_free_rate=0.0,
        trades_per_year=12.0,
    )
    assert (
        intervals["mean_trade_return_ci_low"]
        <= float(series.mean())
        <= intervals["mean_trade_return_ci_high"]
    )
    assert intervals["annualized_sharpe_ci_low"] <= (
        intervals["annualized_sharpe_ci_high"]
    )


def test_intervals_flag_insufficient_sample():
    series = pd.Series([0.01])
    intervals = block_bootstrap_intervals(
        series,
        block_size=5,
        n_samples=100,
        seed=1,
    )
    assert intervals["insufficient_sample"] is True
    assert math.isnan(intervals["annualized_sharpe_ci_low"])
