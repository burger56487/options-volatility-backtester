"""Feature-timing metadata sidecars.

DataFrame attrs are lost when a frame is written to CSV, so feature tables
are accompanied by a JSON sidecar that survives persistence. Consumers must
load and verify the sidecar before using the features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = [
    "dataset",
    "decision_frequency",
    "execution_time",
    "feature_availability_rule",
    "lag_trading_days",
]


DEFAULT_FEATURE_TIMING_METADATA: dict[str, Any] = {
    "dataset": "volatility_features",
    "decision_frequency": "daily",
    "execution_time": "next_open",
    "feature_availability_rule": "previous_close",
    "lag_trading_days": 1,
}


def save_feature_metadata(
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = metadata or dict(DEFAULT_FEATURE_TIMING_METADATA)
    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"Feature metadata missing required keys: {missing}"
        )
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path


def load_feature_metadata(
    path: str | Path,
) -> dict[str, Any]:
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Feature metadata sidecar not found: {metadata_path}"
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    missing = [key for key in REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ValueError(
            f"Feature metadata missing required keys: {missing}"
        )
    return metadata
