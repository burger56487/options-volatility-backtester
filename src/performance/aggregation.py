"""Multi-seed / multi-run metric aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_metric_runs(
    runs: pd.DataFrame,
    metric_column: str,
) -> dict:
    """Report mean/median/std across independent runs of one metric."""
    values = runs[metric_column].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {
            "metric": metric_column,
            "n_runs": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }
    return {
        "metric": metric_column,
        "n_runs": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1))
        if values.size > 1
        else 0.0,
    }
