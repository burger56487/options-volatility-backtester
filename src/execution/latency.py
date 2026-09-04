"""Deterministic execution-latency sampling."""

from __future__ import annotations

import numpy as np


def sample_latency_seconds(
    base_seconds: float,
    jitter_seconds: float,
    seed: int | None = None,
) -> float:
    """Sample latency from Uniform(base, base + jitter) deterministically."""
    if base_seconds < 0 or jitter_seconds < 0:
        raise ValueError("Latency components cannot be negative.")
    rng = np.random.default_rng(seed)
    return float(base_seconds + rng.uniform(0.0, jitter_seconds))
