from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path


import numpy as np
import pandas as pd
from src.backtest.volatility_filter import (
    VolatilityRegimeFilter,
    filter_entry_dates,
)
from src.backtest.timeline import filter_entry_dates_lagged


from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
    run_long_straddle_backtest,
)
from src.market_data.synthetic_option_chain import (
    VolatilitySurfaceParameters,
)
from src.experiment_meta import experiment_metadata
from src.statistics import block_bootstrap_intervals


@dataclass(frozen=True)
class RollingBacktestConfig:
    """
    Configuration for a sequence of non-overlapping long-straddle trades.

    entry_spacing_trading_days:
        Number of trading days between candidate entry dates. For a non-overlap
        design, this should normally be at least days_to_expiry in the underlying
        single-trade configuration.
    """

    entry_spacing_trading_days: int = 30
    initial_capital: float = 100_000.0
    risk_free_rate_for_sharpe: float = 0.0
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.entry_spacing_trading_days < 1:
            raise ValueError(
                "entry_spacing_trading_days must be at least 1."
            )

        if self.initial_capital <= 0:
            raise ValueError(
                "initial_capital must be positive."
            )

        if not 0 < self.confidence_level < 1:
            raise ValueError(
                "confidence_level must be between 0 and 1."
            )


@dataclass(frozen=True)
class RollingBacktestResult:
    """Aggregated output from multiple non-overlapping option trades."""

    trade_results: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: dict[str, float | int | str]


def _validate_price_data(
    price_data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate basic price data requirements for rolling backtesting."""
    if price_data.empty:
        raise ValueError("price_data must not be empty.")

    if "Close" not in price_data.columns:
        raise ValueError(
            "price_data must include a Close column."
        )

    if not isinstance(price_data.index, pd.DatetimeIndex):
        raise TypeError(
            "price_data index must be a DatetimeIndex."
        )

    if (price_data["Close"].dropna() <= 0).any():
        raise ValueError("Close prices must be positive.")

    result = price_data.copy()
    result = result[~result.index.duplicated(keep="last")]
    return result.sort_index()


def _candidate_entry_dates(
    price_data: pd.DataFrame,
    minimum_history_days: int,
    entry_spacing_trading_days: int,
    days_to_expiry: int,
) -> list[pd.Timestamp]:
    """
    Generate candidate non-overlapping entry dates.

    An entry requires:
    - enough historical rows for realised-volatility estimation;
    - enough future rows to approximately cover the target option maturity.
    """
    start_index = minimum_history_days
    last_entry_index = (
        len(price_data)
        - days_to_expiry
        - 1
    )

    if last_entry_index < start_index:
        return []

    return [
        pd.Timestamp(price_data.index[index])
        for index in range(
            start_index,
            last_entry_index + 1,
            entry_spacing_trading_days,
        )
    ]


def _risk_summary(
    trade_returns: pd.Series,
    confidence_level: float,
    risk_free_rate_for_sharpe: float,
) -> dict[str, float]:
    """Calculate trade-level risk and performance statistics."""
    clean_returns = trade_returns.dropna()

    if clean_returns.empty:
        return {
            "mean_trade_return": 0.0,
            "trade_return_volatility": 0.0,
            "sharpe_like_ratio": 0.0,
            "var": 0.0,
            "expected_shortfall": 0.0,
        }

    mean_return = float(clean_returns.mean())
    volatility = float(clean_returns.std(ddof=1))

    if len(clean_returns) < 2 or volatility == 0:
        sharpe_like_ratio = 0.0
    else:
        sharpe_like_ratio = (
            mean_return - risk_free_rate_for_sharpe
        ) / volatility * sqrt(len(clean_returns))

    loss_quantile = float(
        clean_returns.quantile(1.0 - confidence_level)
    )

    var = max(0.0, -loss_quantile)

    tail_returns = clean_returns[
        clean_returns <= loss_quantile
    ]

    expected_shortfall = (
        0.0
        if tail_returns.empty
        else max(0.0, -float(tail_returns.mean()))
    )

    return {
        "mean_trade_return": mean_return,
        "trade_return_volatility": volatility,
        "sharpe_like_ratio": sharpe_like_ratio,
        "legacy_sharpe_like_ratio": sharpe_like_ratio,
        "var": var,
        "expected_shortfall": expected_shortfall,
    }


def run_rolling_long_straddle_backtest(
    price_data: pd.DataFrame,
    trade_config: LongStraddleBacktestConfig = (
        LongStraddleBacktestConfig()
    ),
    rolling_config: RollingBacktestConfig = (
        RollingBacktestConfig()
    ),
    surface_parameters: VolatilitySurfaceParameters = (
        VolatilitySurfaceParameters()
    ),
    regime_filter: VolatilityRegimeFilter | None = None,
    lag_regime_signal: bool = True,
) -> RollingBacktestResult:
    """
    Run a sequence of non-overlapping delta-hedged long-straddle backtests.

    The result is a trade-level study rather than a fully capital-constrained
    portfolio simulator. Every trade uses the same contract quantity and its
    P&L is added to an illustrative initial-capital equity curve.
    """
    data = _validate_price_data(price_data)

    minimum_history_days = max(
        trade_config.volatility_windows
    )

    entry_dates = _candidate_entry_dates(
        price_data=data,
        minimum_history_days=minimum_history_days,
        entry_spacing_trading_days=(
            rolling_config.entry_spacing_trading_days
        ),
        days_to_expiry=trade_config.days_to_expiry,
    )
    candidate_entry_dates = entry_dates.copy()


    if not entry_dates:
        raise ValueError(
            "Insufficient data for rolling backtest."
        )
        candidate_entry_dates = entry_dates

    if regime_filter is not None:
        filter_function = (
            filter_entry_dates_lagged
            if lag_regime_signal
            else filter_entry_dates
        )
        filter_results = filter_function(
            price_data=data,
            candidate_dates=candidate_entry_dates,
            regime_filter=regime_filter,
        )

        entry_dates = [
            pd.Timestamp(date)
            for date in filter_results.index[
                filter_results["selected"]
            ]
        ]

    if not entry_dates:
        raise ValueError(
            "No entry dates passed the volatility-regime filter."
        )

    records: list[dict[str, float | int | str]] = []

    capital = rolling_config.initial_capital

    for entry_date in entry_dates:
        result = run_long_straddle_backtest(
            price_data=data,
            entry_date=entry_date,
            config=trade_config,
            surface_parameters=surface_parameters,
        )

        summary = result.summary
        trade_pnl = float(summary["final_pnl"])
        entry_cost = float(summary["entry_cost"])
        trade_return = trade_pnl / entry_cost

        capital += trade_pnl

        records.append(
            {
                "entry_date": str(summary["entry_date"]),
                "expiry_date": str(summary["expiry_date"]),
                "entry_spot": float(summary["entry_spot"]),
                "final_spot": float(summary["final_spot"]),
                "strike": float(summary["strike"]),
                "entry_cost": entry_cost,
                "final_pnl": trade_pnl,
                "trade_return": trade_return,
                "max_drawdown": float(
                    summary["max_drawdown"]
                ),
                "number_of_hedge_trades": int(
                    summary["number_of_hedge_trades"]
                ),
                "cumulative_hedge_turnover": float(
                    summary["cumulative_hedge_turnover"]
                ),
                "cumulative_hedge_costs": float(
                    summary["cumulative_hedge_costs"]
                ),
                "hedge_turnover_ratio": float(
                    summary["hedge_turnover_ratio"]
                ),
                "capital_after_trade": capital,
            }
        )

    trade_results = pd.DataFrame(records)
    trade_results["entry_date"] = pd.to_datetime(
        trade_results["entry_date"]
    )
    trade_results["expiry_date"] = pd.to_datetime(
        trade_results["expiry_date"]
    )

    trade_results = trade_results.sort_values(
        "entry_date"
    ).reset_index(drop=True)

    equity_curve = trade_results[
        [
            "entry_date",
            "expiry_date",
            "final_pnl",
            "trade_return",
            "capital_after_trade",
        ]
    ].copy()

    equity_curve = equity_curve.rename(
        columns={
            "entry_date": "date",
            "capital_after_trade": "equity",
        }
    )

    equity_curve = equity_curve.set_index("date")

    running_peak = equity_curve["equity"].cummax()
    equity_curve["drawdown"] = (
        equity_curve["equity"] - running_peak
    )
    equity_curve["drawdown_pct"] = (
        equity_curve["equity"] / running_peak - 1.0
    )

    risk_metrics = _risk_summary(
        trade_returns=trade_results["trade_return"],
        confidence_level=rolling_config.confidence_level,
        risk_free_rate_for_sharpe=(
            rolling_config.risk_free_rate_for_sharpe
        ),
    )

    trades_per_year = max(
        1.0,
        252.0 / float(rolling_config.entry_spacing_trading_days),
    )

    stat_intervals = block_bootstrap_intervals(
        trade_returns=trade_results["trade_return"],
        block_size=5,
        n_samples=2_000,
        seed=20260814,
        confidence_level=rolling_config.confidence_level,
        risk_free_rate=(
            rolling_config.risk_free_rate_for_sharpe
        ),
        trades_per_year=trades_per_year,
    )

    total_pnl = float(trade_results["final_pnl"].sum())
    win_rate = float(
        (trade_results["final_pnl"] > 0).mean()
    )

    summary: dict[str, float | int | str] = {
        "candidate_entry_dates": int(len(candidate_entry_dates)),
        "selected_entry_dates": int(len(entry_dates)),
        "regime_filter_applied": str(regime_filter is not None),
        "regime_signal_lagged": str(lag_regime_signal),
        "number_of_trades": int(len(trade_results)),

        "final_capital": float(
            equity_curve["equity"].iloc[-1]
        ),
        "total_pnl": total_pnl,
        "total_return_on_initial_capital": (
            total_pnl / rolling_config.initial_capital
        ),
        "win_rate": win_rate,
        "average_trade_pnl": float(
            trade_results["final_pnl"].mean()
        ),
        "median_trade_pnl": float(
            trade_results["final_pnl"].median()
        ),
        "max_trade_drawdown": float(
            trade_results["max_drawdown"].min()
        ),
        "portfolio_max_drawdown": float(
            equity_curve["drawdown"].min()
        ),
        "portfolio_max_drawdown_pct": float(
            equity_curve["drawdown_pct"].min()
        ),
        "average_hedge_trades_per_option_trade": float(
            trade_results["number_of_hedge_trades"].mean()
        ),
        "total_hedge_turnover": float(
            trade_results[
                "cumulative_hedge_turnover"
            ].sum()
        ),
        "total_hedge_costs": float(
            trade_results[
                "cumulative_hedge_costs"
            ].sum()
        ),
        "average_hedge_turnover_ratio": float(
            trade_results["hedge_turnover_ratio"].mean()
        ),
        **risk_metrics,
        **stat_intervals,
        "trades_per_year_assumed": trades_per_year,
        "metric_definitions": (
            "sharpe_like_ratio = (mean_trade_return - risk_free) / "
            "trade_return_volatility * sqrt(n_trades); "
            "annualized_sharpe_estimate = trade-level Sharpe * "
            "sqrt(trades_per_year_assumed); "
            "confidence intervals use a moving-block bootstrap "
            "of trade returns."
        ),
        **experiment_metadata(),
    }

    return RollingBacktestResult(
        trade_results=trade_results,
        equity_curve=equity_curve,
        summary=summary,
    )


def save_rolling_backtest_result(
    result: RollingBacktestResult,
    output_directory: str | Path,
    prefix: str = "rolling_long_straddle",
) -> None:
    """Save rolling trade results, equity curve, and summary."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    result.trade_results.to_csv(
        output_path / f"{prefix}_trade_results.csv",
        index=False,
    )

    result.equity_curve.to_csv(
        output_path / f"{prefix}_equity_curve.csv",
    )

    pd.DataFrame([result.summary]).to_json(
        output_path / f"{prefix}_summary.json",
        orient="records",
        indent=2,
    )
