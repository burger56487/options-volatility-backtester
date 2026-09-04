"""Per-run output directory management with boundary metadata."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .run_metadata import create_run_metadata, save_run_metadata


@dataclass
class RunContext:
    run_id: str
    output_directory: Path
    metadata: dict[str, Any]
    config: dict[str, Any]


def initialise_run(
    config: dict[str, Any],
    config_path: str | Path,
    command: str | None = None,
) -> RunContext:
    """Create a run-specific output directory with metadata snapshots."""
    metadata = create_run_metadata(config=config, command=command)
    run_id = metadata["run_id"]

    base_output_directory = Path(config["output"]["directory"])
    run_output_directory = base_output_directory / run_id
    run_output_directory.mkdir(parents=True, exist_ok=False)

    if config["output"].get("save_metadata", True):
        save_run_metadata(
            metadata=metadata,
            output_directory=run_output_directory,
        )

    if config["output"].get("save_config_snapshot", True):
        shutil.copy2(
            config_path,
            run_output_directory / "config_snapshot.yaml",
        )

    boundary = {
        "underlying_data": (
            config["data"]["underlying"]["data_type"]
        ),
        "option_data": (
            config["data"]["options"]["data_type"]
        ),
        "execution": config["execution"]["mode"],
        "evaluation": config["research"]["evaluation_mode"],
        "is_real_option_backtest": (
            config["data"]["options"]["data_type"] == "real"
            and config["execution"]["mode"] in {"paper", "live"}
        ),
    }
    boundary_path = run_output_directory / "research_boundary.json"
    with boundary_path.open("w", encoding="utf-8") as file:
        json.dump(
            boundary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return RunContext(
        run_id=run_id,
        output_directory=run_output_directory,
        metadata=metadata,
        config=config,
    )
