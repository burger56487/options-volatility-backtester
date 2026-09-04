"""Both moving-block bootstrap surfaces share one sampler implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.performance.bootstrap import moving_block_samples
from src.statistics import (
    block_size_heuristic,
    moving_block_bootstrap_samples,
)


def _series(size: int = 50) -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0, 1.0, size=size))


def test_samplers_are_identical_given_block_size():
    series = _series()
    performance = moving_block_samples(
        series,
        n_samples=400,
        block_size=4,
        seed=9,
    )
    statistics = moving_block_bootstrap_samples(
        series,
        block_size=4,
        n_samples=400,
        seed=9,
    )
    assert performance.shape == statistics.shape
    assert np.array_equal(performance, statistics)


def test_performance_default_matches_shared_heuristic():
    series = _series(size=30)
    default = moving_block_samples(
        series,
        n_samples=200,
        seed=3,
    )
    expected = moving_block_bootstrap_samples(
        series,
        block_size=block_size_heuristic(30),
        n_samples=200,
        seed=3,
    )
    assert np.array_equal(default, expected)


def test_heuristic_values():
    assert block_size_heuristic(1) == 1
    assert block_size_heuristic(8) == 2
    assert block_size_heuristic(30) == 4
