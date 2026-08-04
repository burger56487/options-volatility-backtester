from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
    run_long_straddle_backtest,
)
from src.market_data.synthetic_option_chain import (
    VolatilitySurfaceParameters,
)


def run_hedging_sensitivity_analysis(
    price_data: pd.DataFrame,
    entry_date: pd.Timestamp,
    base_config: LongStraddleBacktestConfig,
    delta_thresholds: tuple[float, ...] = (
        0.0,
        1.0,
        5.0,
        10.0,
        20.0,
    ),
    slippage_bps_values: tuple[float, ...] = (
        0.5,
        1.0,
        5.0,
        10.0,
    ),
    surface_parameters: VolatilitySurfaceParameters = (
        VolatilitySurfaceParameters()
    ),
) -> pd.DataFrame:
    """
    Run a grid of delta-hedging threshold and slippage scenarios.

    Each row represents one backtest configuration. The function returns
    summary statistics and does not mutate the input base configuration.
    """
    if not delta_thresholds:
        raise ValueError(
            "delta_thresholds must not be empty."
        )

    if not slippage_bps_values:
        raise ValueError(
            "slippage_bps_values must not be empty."
        )

    if any(threshold < 0 for threshold in delta_thresholds):
        raise ValueError(
            "delta_thresholds must be non-negative."
        )

    if any(slippage < 0 for slippage in slippage_bps_values):
        raise ValueError(
            "slippage_bps_values must be non-negative."
        )

    records: list[dict[str, float | int | str]] = []

    for delta_threshold in delta_thresholds:
        for slippage_bps in slippage_bps_values:
            scenario_config = replace(
                base_config,
                delta_threshold=delta_threshold,
                underlying_slippage_bps=slippage_bps,
            )

            result = run_long_straddle_backtest(
                price_data=price_data,
                entry_date=entry_date,
                config=scenario_config,
                surface_parameters=surface_parameters,
            )

            summary = result.summary

            records.append(
                {
                    "delta_threshold": delta_threshold,
                    "slippage_bps": slippage_bps,
                    "final_pnl": float(summary["final_pnl"]),
                    "max_drawdown": float(
                        summary["max_drawdown"]
                    ),
                    "number_of_hedge_trades": int(
                        summary["number_of_hedge_trades"]
                    ),
                    "cumulative_hedge_turnover": float(
                        summary[
                            "cumulative_hedge_turnover"
                        ]
                    ),
                    "cumulative_hedge_costs": float(
                        summary["cumulative_hedge_costs"]
                    ),
                    "hedge_turnover_ratio": float(
                        summary["hedge_turnover_ratio"]
                    ),
                    "entry_cost": float(summary["entry_cost"]),
                    "final_option_value": float(
                        summary["final_option_value"]
                    ),
                    "final_hedge_equity": float(
                        summary["final_hedge_equity"]
                    ),
                }
            )

    return pd.DataFrame(records).sort_values(
        ["delta_threshold", "slippage_bps"]
    ).reset_index(drop=True)


def create_sensitivity_pivot(
    results: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """
    Pivot sensitivity results into threshold-by-slippage matrix form.

    Rows are delta thresholds and columns are slippage assumptions.
    """
    required_columns = {
        "delta_threshold",
        "slippage_bps",
        value_column,
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return results.pivot(
        index="delta_threshold",
        columns="slippage_bps",
        values=value_column,
    ).sort_index().sort_index(axis=1)
