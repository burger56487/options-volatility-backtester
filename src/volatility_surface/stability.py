"""Cross-date parameter stability reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def parameter_stability(
    calibrations: pd.DataFrame,
    parameter_columns: list[str],
) -> pd.DataFrame:
    """Report std / range across dates for each SVI parameter."""
    rows = []
    for column in parameter_columns:
        if column not in calibrations.columns:
            continue
        values = calibrations[column].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        rows.append(
            {
                "parameter": column,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1))
                if values.size > 1
                else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "range": float(np.max(values) - np.min(values)),
            }
        )
    return pd.DataFrame(rows)
