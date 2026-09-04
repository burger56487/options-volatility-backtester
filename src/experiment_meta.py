"""Standard experiment metadata attached to every backtest result.

Every saved result must be able to identify, without reading the README,
what kind of data and execution assumptions produced it. This module builds
that metadata in one place so synthetic and real-market experiments cannot
be confused downstream.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone


def experiment_metadata(
    option_data_type: str = "synthetic",
    strategy_version: str = "1.1.0",
) -> dict[str, str]:
    """Return standard metadata for a backtest result."""
    return {
        "underlying_data_type": "real",
        "option_data_type": option_data_type,
        "execution_type": "simulated",
        "strategy_version": strategy_version,
        "git_commit": os.environ.get(
            "GITHUB_SHA",
            "not-recorded",
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
    }
