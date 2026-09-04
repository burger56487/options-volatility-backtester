"""YAML-based research configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def load_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate a YAML research configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration top level must be a mapping.")

    validate_config(config)
    return deepcopy(config)


def validate_config(config: dict[str, Any]) -> None:
    """Validate required sections and value domains of a configuration."""
    required_sections = [
        "project",
        "data",
        "execution",
        "research",
        "output",
    ]
    missing_sections = [
        section
        for section in required_sections
        if section not in config
    ]
    if missing_sections:
        raise ValueError(
            "Configuration missing required sections: "
            f"{', '.join(missing_sections)}"
        )

    option_data_type = (
        config["data"].get("options", {}).get("data_type")
    )
    if option_data_type not in {"real", "synthetic", "mixed"}:
        raise ValueError(
            "data.options.data_type must be real, synthetic or mixed."
        )

    execution_mode = config["execution"].get("mode")
    if execution_mode not in {"simulated", "paper", "live"}:
        raise ValueError(
            "execution.mode must be simulated, paper or live."
        )

    evaluation_mode = config["research"].get("evaluation_mode")
    if evaluation_mode not in {
        "in_sample",
        "validation",
        "out_of_sample",
        "walk_forward",
    }:
        raise ValueError("research.evaluation_mode has an invalid value.")

    signal_delay_days = config["research"].get("signal_delay_days", 0)
    if not isinstance(signal_delay_days, int) or signal_delay_days < 0:
        raise ValueError(
            "research.signal_delay_days must be a non-negative integer."
        )
