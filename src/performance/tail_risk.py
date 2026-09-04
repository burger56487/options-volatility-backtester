"""Tail-risk metrics with a minimum-observation guard."""

from __future__ import annotations

import numpy as np


MINIMUM_OBSERVATIONS = 20


def historical_var_cvar(
    returns,
    confidence_level: float = 0.95,
) -> dict:
    clean = np.asarray(returns.dropna(), dtype=float)
    if clean.size < MINIMUM_OBSERVATIONS:
        return {
            "insufficient_sample": True,
            "observations": int(clean.size),
            "var": float("nan"),
            "cvar": float("nan"),
        }
    quantile = np.quantile(clean, 1.0 - confidence_level)
    tail = clean[clean <= quantile]
    return {
        "insufficient_sample": False,
        "observations": int(clean.size),
        "var": float(max(0.0, -quantile)),
        "cvar": float(max(0.0, -tail.mean())) if tail.size else 0.0,
    }
