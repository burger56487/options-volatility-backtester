from __future__ import annotations

import pandas as pd

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
    run_rolling_long_straddle_backtest,
)
from src.backtest.volatility_filter import (
    VolatilityRegimeFilter,
)
from src.market_data.synthetic_option_chain import (
    VolatilitySurfaceParameters,
)


def run_regime_threshold_sensitivity(
    price_data: pd.DataFrame,
    trade_config: LongStraddleBacktestConfig,
    rolling_config: RollingBacktestConfig,
    thresholds: tuple[float, ...] = (
        0.90,
        1.00,
        1.05,
        1.10,
        1.20,
        1.30,
    ),
    short_window: int = 20,
    long_window: int = 252,
    surface_parameters: VolatilitySurfaceParameters = (
        VolatilitySurfaceParameters()
    ),
) -> pd.DataFrame:
    """
    Run rolling long-straddle backtests across volatility-ratio thresholds.

    A higher threshold requires greater short-term realised volatility relative
    to long-term realised volatility before a trade is opened.
    """
    if not thresholds:
        raise ValueError("thresholds must not be empty.")

    if any(threshold <= 0 for threshold in thresholds):
        raise ValueError("thresholds must be positive.")

    if len(set(thresholds)) != len(thresholds):
        raise ValueError("thresholds must not contain duplicates.")

    records: list[dict[str, float | int | str]] = []

    for threshold in sorted(thresholds):
        regime_filter = VolatilityRegimeFilter(
            short_window=short_window,
            long_window=long_window,
            minimum_volatility_ratio=threshold,
        )

        result = run_rolling_long_straddle_backtest(
            price_data=price_data,
            trade_config=trade_config,
            rolling_config=rolling_config,
            surface_parameters=surface_parameters,
            regime_filter=regime_filter,
        )

        summary = result.summary

        records.append(
            {
                "volatility_ratio_threshold": threshold,
                "number_of_trades": int(
                    summary["number_of_trades"]
                ),
                "total_pnl": float(summary["total_pnl"]),
                "total_return_on_initial_capital": float(
                    summary[
                        "total_return_on_initial_capital"
                    ]
                ),
                "win_rate": float(summary["win_rate"]),
                "average_trade_pnl": float(
                    summary["average_trade_pnl"]
                ),
                "median_trade_pnl": float(
                    summary["median_trade_pnl"]
                ),
                "portfolio_max_drawdown": float(
                    summary["portfolio_max_drawdown"]
                ),
                "portfolio_max_drawdown_pct": float(
                    summary[
                        "portfolio_max_drawdown_pct"
                    ]
                ),
                "sharpe_like_ratio": float(
                    summary["sharpe_like_ratio"]
                ),
                "var": float(summary["var"]),
                "expected_shortfall": float(
                    summary["expected_shortfall"]
                ),
                "total_hedge_turnover": float(
                    summary["total_hedge_turnover"]
                ),
                "total_hedge_costs": float(
                    summary["total_hedge_costs"]
                ),
                "average_hedge_turnover_ratio": float(
                    summary[
                        "average_hedge_turnover_ratio"
                    ]
                ),
                "candidate_entry_dates": int(
                    summary["candidate_entry_dates"]
                ),
                "selected_entry_dates": int(
                    summary["selected_entry_dates"]
                ),
            }
        )

    return pd.DataFrame(records).sort_values(
        "volatility_ratio_threshold"
    ).reset_index(drop=True)
