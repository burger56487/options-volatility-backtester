"""Parameter selection restricted to train and validation data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass
class CandidateResult:
    parameters: dict[str, Any]
    train_score: float
    validation_score: float
    train_trades: int
    validation_trades: int


def calculate_selection_score(
    sharpe: float,
    max_drawdown: float,
    turnover: float,
    trade_count: int,
    drawdown_penalty: float = 1.0,
    turnover_penalty: float = 0.01,
    small_sample_penalty: float = 1.0,
) -> float:
    if trade_count <= 0:
        return float("-inf")
    return (
        sharpe
        - drawdown_penalty * abs(max_drawdown)
        - turnover_penalty * turnover
        - small_sample_penalty / math.sqrt(trade_count)
    )


def select_parameters(
    candidates: list[dict[str, Any]],
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    evaluator: Callable[
        [dict[str, Any], pd.DataFrame],
        dict[str, float],
    ],
    minimum_validation_trades: int = 10,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select parameters using validation score only."""
    results = []
    for parameters in candidates:
        train_metrics = evaluator(parameters, train_data)
        validation_metrics = evaluator(parameters, validation_data)
        results.append(
            CandidateResult(
                parameters=parameters,
                train_score=float(train_metrics["selection_score"]),
                validation_score=float(
                    validation_metrics["selection_score"]
                ),
                train_trades=int(train_metrics["trade_count"]),
                validation_trades=int(
                    validation_metrics["trade_count"]
                ),
            )
        )

    eligible = [
        result
        for result in results
        if result.validation_trades >= minimum_validation_trades
    ]
    if not eligible:
        raise ValueError(
            "No candidate satisfies the minimum validation trade count."
        )
    selected = max(
        eligible,
        key=lambda result: result.validation_score,
    )
    report = pd.DataFrame(
        [
            {
                **result.parameters,
                "train_score": result.train_score,
                "validation_score": result.validation_score,
                "train_trades": result.train_trades,
                "validation_trades": result.validation_trades,
                "selected": result.parameters == selected.parameters,
            }
            for result in results
        ]
    )
    return selected.parameters, report
