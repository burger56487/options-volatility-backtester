"""Out-of-sample evaluation harness for the volatility-regime filter.

The regime-filter threshold is selected on a training window only, then the
selected threshold is evaluated once on a held-out test window. This prevents
choosing the best-looking threshold on the full sample and presenting it as
out-of-sample evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
    RollingBacktestResult,
    run_rolling_long_straddle_backtest,
)
from src.backtest.volatility_filter import (
    VolatilityRegimeFilter,
)


@dataclass(frozen=True)
class OosFilterResult:
    """Artifacts of one out-of-sample filter evaluation."""

    selected_threshold: float
    threshold_scores: list[tuple[float, float, int]]
    train_result: RollingBacktestResult
    test_result: RollingBacktestResult
    baseline_test_result: RollingBacktestResult


def chronological_split(
    price_data: pd.DataFrame,
    train_fraction: float = 0.6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split price data chronologically into train and test windows."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    split_index = int(len(price_data) * train_fraction)
    return (
        price_data.iloc[:split_index].copy(),
        price_data.iloc[split_index:].copy(),
    )


def _score_threshold(
    price_data: pd.DataFrame,
    threshold: float,
    trade_config: LongStraddleBacktestConfig,
    rolling_config: RollingBacktestConfig,
    metric: str,
) -> tuple[float, int]:
    try:
        result = run_rolling_long_straddle_backtest(
            price_data=price_data,
            trade_config=trade_config,
            rolling_config=rolling_config,
            regime_filter=VolatilityRegimeFilter(
                minimum_volatility_ratio=threshold
            ),
        )
        score = float(result.summary.get(metric, float("-inf")))
        return score, int(result.summary["number_of_trades"])
    except ValueError:
        return float("-inf"), 0


def select_threshold_on_train(
    train_data: pd.DataFrame,
    candidate_thresholds: list[float],
    trade_config: LongStraddleBacktestConfig,
    rolling_config: RollingBacktestConfig,
    metric: str = "annualized_sharpe_estimate",
) -> tuple[float, list[tuple[float, float, int]]]:
    """Select the best threshold using training-window results only."""
    scores: list[tuple[float, float, int]] = []
    for threshold in candidate_thresholds:
        score, trades = _score_threshold(
            price_data=train_data,
            threshold=threshold,
            trade_config=trade_config,
            rolling_config=rolling_config,
            metric=metric,
        )
        scores.append((threshold, score, trades))

    selected = max(
        scores,
        key=lambda item: (
            item[1],
            item[2],  # tie-break towards more trades
        ),
    )
    return selected[0], scores


def evaluate_filter_out_of_sample(
    price_data: pd.DataFrame,
    candidate_thresholds: list[float],
    trade_config: LongStraddleBacktestConfig = (
        LongStraddleBacktestConfig(days_to_expiry=30, delta_threshold=5.0)
    ),
    rolling_config: RollingBacktestConfig = RollingBacktestConfig(
        entry_spacing_trading_days=30,
        initial_capital=100_000.0,
    ),
    train_fraction: float = 0.6,
    metric: str = "annualized_sharpe_estimate",
) -> OosFilterResult:
    """Run the full train-select / test-evaluate protocol."""
    train_data, test_data = chronological_split(
        price_data=price_data,
        train_fraction=train_fraction,
    )

    selected_threshold, scores = select_threshold_on_train(
        train_data=train_data,
        candidate_thresholds=candidate_thresholds,
        trade_config=trade_config,
        rolling_config=rolling_config,
        metric=metric,
    )

    train_result = run_rolling_long_straddle_backtest(
        price_data=train_data,
        trade_config=trade_config,
        rolling_config=rolling_config,
        regime_filter=VolatilityRegimeFilter(
            minimum_volatility_ratio=selected_threshold
        ),
    )
    test_result = run_rolling_long_straddle_backtest(
        price_data=test_data,
        trade_config=trade_config,
        rolling_config=rolling_config,
        regime_filter=VolatilityRegimeFilter(
            minimum_volatility_ratio=selected_threshold
        ),
    )
    baseline_test_result = run_rolling_long_straddle_backtest(
        price_data=test_data,
        trade_config=trade_config,
        rolling_config=rolling_config,
    )

    return OosFilterResult(
        selected_threshold=selected_threshold,
        threshold_scores=scores,
        train_result=train_result,
        test_result=test_result,
        baseline_test_result=baseline_test_result,
    )
