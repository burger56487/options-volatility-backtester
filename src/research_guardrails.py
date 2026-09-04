"""Guardrails preventing misleading data-boundary claims."""

from __future__ import annotations

from typing import Any


FORBIDDEN_REAL_CLAIMS = [
    "真实期权回测",
    "真实期权市场回测",
    "实盘",
    "live trading",
    "real option backtest",
]


def validate_research_claims(
    config: dict[str, Any],
    run_label: str,
) -> None:
    """Reject labels that claim real trading on synthetic data."""
    option_data_type = config["data"]["options"]["data_type"]
    execution_mode = config["execution"]["mode"]
    lower_label = run_label.lower()

    if option_data_type != "real":
        for phrase in FORBIDDEN_REAL_CLAIMS:
            if phrase.lower() in lower_label:
                raise ValueError(
                    "Run label conflicts with data boundary: option data "
                    "is not real historical option quotes."
                )

    if execution_mode == "simulated" and "实盘" in run_label:
        raise ValueError(
            "Run label conflicts with execution mode: execution is simulated."
        )
