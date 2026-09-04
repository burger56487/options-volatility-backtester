"""Import existing output directories into the repository."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .repository import RunRepository


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {}


def import_run_directory(
    repository: RunRepository,
    run_directory: str | Path,
    run_id: str,
) -> None:
    """Persist one run directory: metadata, boundary, metrics, summary."""
    directory = Path(run_directory)
    metadata = _read_json(directory / "run_metadata.json")
    boundary = _read_json(directory / "research_boundary.json")
    metrics = _read_json(directory / "summary.json")
    merged_metrics = dict(metrics)
    if boundary:
        merged_metrics["research_boundary"] = boundary
    repository.save_run(
        run_id=run_id,
        created_at=metadata.get(
            "created_at_utc",
            metadata.get("run_id", run_id),
        ),
        metadata=metadata,
        metrics=merged_metrics,
    )
