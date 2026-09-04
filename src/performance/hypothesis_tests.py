"""Paired daily-difference bootstrap test between two strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import moving_block_samples


def paired_daily_difference_p_value(
    strategy_a: pd.Series,
    strategy_b: pd.Series,
    n_samples: int = 2_000,
    seed: int | None = None,
) -> dict:
    aligned = pd.concat([strategy_a, strategy_b], axis=1).dropna()
    if len(aligned) < 5:
        return {"insufficient_sample": True, "p_value": float("nan")}
    difference = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    observed = float(difference.mean())
    samples = moving_block_samples(
        difference,
        n_samples=n_samples,
        seed=seed,
    )
    means = samples.mean(axis=1)
    p_value = float(
        (np.abs(means) >= abs(observed)).mean()
    )
    return {"insufficient_sample": False, "p_value": p_value}
