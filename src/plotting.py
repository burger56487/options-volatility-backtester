"""Plot helpers that make research boundaries visible on figures."""

from __future__ import annotations

from typing import Any


def build_research_subtitle(config: dict[str, Any]) -> str:
    """Build a figure subtitle describing the data and execution boundary."""
    underlying = config["data"]["underlying"]["data_type"]
    options = config["data"]["options"]["data_type"]
    execution = config["execution"]["mode"]
    evaluation = config["research"]["evaluation_mode"]
    return (
        f"underlying={underlying} | options={options} | "
        f"execution={execution} | evaluation={evaluation}"
    )
