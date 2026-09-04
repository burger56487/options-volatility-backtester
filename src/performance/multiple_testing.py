"""Multiple-comparison awareness for parameter searches."""

from __future__ import annotations

import numpy as np


def bonferroni_threshold(alpha: float, n_trials: int) -> float:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    return alpha / n_trials


def max_t_p_value(
    candidate_samples: np.ndarray,
    observed_max: float,
) -> float:
    """Fraction of bootstrap draws where the best candidate beats observed."""
    maxima = candidate_samples.max(axis=0)
    return float((maxima >= observed_max).mean())
