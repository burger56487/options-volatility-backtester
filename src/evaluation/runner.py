"""Real strict-evaluation runner wired to the rolling backtest engine.

The runner executes one full rolling backtest per candidate threshold (with
the regime signal lagged by one day), labels every trade by its entry date,
and reports train / validation metrics for parameter selection. The selected
threshold is evaluated once on the locked test split. Train and validation
trades that would exit after their split boundary are excluded from
parameter selection and reported explicitly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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
from src.evaluation.model_selection import (
    calculate_selection_score,
)
from src.evaluation.splits import DateSplit, TrainValidationTestSplit
from src.evaluation.test_lock import (
    check_test_evaluation_allowed,
    record_test_evaluation,
)


def label_trades_by_split(
    trades: pd.DataFrame,
    split: TrainValidationTestSplit,
) -> pd.DataFrame:
    labelled = trades.copy()
    labelled["split_name"] = labelled["entry_date"].apply(split.locate)
    return labelled


def boundary_crossing_trades(
    labelled_trades: pd.DataFrame,
    split: DateSplit,
) -> pd.DataFrame:
    """Trades entered in the split that exit after its end date."""
    if labelled_trades.empty:
        return labelled_trades
    mask = (
        (labelled_trades["split_name"] == split.name)
        & (
            labelled_trades["expiry_date"]
            > pd.Timestamp(split.end_date)
        )
    )
    return labelled_trades[mask].copy()


def split_metrics(
    trades: pd.DataFrame,
    trades_per_year: float,
) -> dict[str, float]:
    if trades.empty:
        return {
            "trade_count": 0.0,
            "total_pnl": 0.0,
            "mean_trade_return": 0.0,
            "annualized_sharpe_estimate": 0.0,
            "selection_score": float("-inf"),
        }
    count = len(trades)
    returns = trades["trade_return"]
    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    annualized = (
        (mean / std) * math.sqrt(trades_per_year)
        if count > 1 and std > 0
        else 0.0
    )
    max_drawdown = abs(float(trades["max_drawdown"].min()))
    turnover = float(trades["hedge_turnover_ratio"].mean())
    return {
        "trade_count": float(count),
        "total_pnl": float(trades["final_pnl"].sum()),
        "mean_trade_return": mean,
        "annualized_sharpe_estimate": annualized,
        "selection_score": calculate_selection_score(
            sharpe=annualized,
            max_drawdown=max_drawdown,
            turnover=turnover,
            trade_count=count,
        ),
    }


def run_strict_evaluation(
    price_data: pd.DataFrame,
    candidate_thresholds: list[float],
    split: TrainValidationTestSplit,
    output_directory: str | Path,
    run_id: str,
    trade_config: LongStraddleBacktestConfig,
    rolling_config: RollingBacktestConfig,
    minimum_validation_trades: int = 6,
    lag_regime_signal: bool = True,
    test_lock_path: str | Path = "evaluation/test_evaluation_log.json",
    allow_test_repeat: bool = False,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Run selection on train/validation and one locked test evaluation."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    trades_per_year = max(
        1.0,
        252.0 / float(rolling_config.entry_spacing_trading_days),
    )

    selection_rows = []
    best = None
    for threshold in candidate_thresholds:
        result = run_rolling_long_straddle_backtest(
            price_data=price_data,
            trade_config=trade_config,
            rolling_config=rolling_config,
            regime_filter=VolatilityRegimeFilter(
                minimum_volatility_ratio=threshold
            ),
            lag_regime_signal=lag_regime_signal,
        )
        labelled = label_trades_by_split(result.trade_results, split)
        train_crossing = boundary_crossing_trades(
            labelled, split.train
        )
        validation_crossing = boundary_crossing_trades(
            labelled, split.validation
        )
        train_trades = labelled[
            labelled["split_name"] == "train"
        ]
        validation_trades = labelled[
            labelled["split_name"] == "validation"
        ]
        train_metrics = split_metrics(train_trades, trades_per_year)
        validation_metrics = split_metrics(
            validation_trades,
            trades_per_year,
        )
        row = {
            "minimum_volatility_ratio": threshold,
            "train_score": train_metrics["selection_score"],
            "validation_score": validation_metrics["selection_score"],
            "train_trades": int(train_metrics["trade_count"]),
            "validation_trades": int(
                validation_metrics["trade_count"]
            ),
            "selected": False,
        }
        selection_rows.append(row)
        if (
            validation_metrics["trade_count"]
            >= minimum_validation_trades
            and (
                best is None
                or validation_metrics["selection_score"]
                > best[1]["selection_score"]
            )
        ):
            best = (
                threshold,
                train_metrics,
                validation_metrics,
                labelled,
                train_crossing,
                validation_crossing,
            )

    if best is None:
        raise ValueError(
            "No threshold meets the minimum validation trade count."
        )
    for row in selection_rows:
        row["selected"] = row[
            "minimum_volatility_ratio"
        ] == best[0]

    (
        selected_threshold,
        best_train_metrics,
        best_validation_metrics,
        labelled_all,
        train_crossing,
        validation_crossing,
    ) = best

    selection_report = pd.DataFrame(selection_rows)
    selection_report.to_csv(
        output_path / "parameter_selection.csv",
        index=False,
    )

    check_test_evaluation_allowed(
        lock_file=test_lock_path,
        allow_repeat=allow_test_repeat,
    )
    test_result = run_rolling_long_straddle_backtest(
        price_data=price_data,
        trade_config=trade_config,
        rolling_config=rolling_config,
        regime_filter=VolatilityRegimeFilter(
            minimum_volatility_ratio=selected_threshold
        ),
        lag_regime_signal=lag_regime_signal,
    )
    test_labelled = label_trades_by_split(
        test_result.trade_results, split
    )
    test_trades = test_labelled[
        test_labelled["split_name"] == "test"
    ]
    test_metrics = split_metrics(test_trades, trades_per_year)

    record_test_evaluation(
        lock_file=test_lock_path,
        run_id=run_id,
        selected_parameters={
            "minimum_volatility_ratio": selected_threshold
        },
        git_commit=git_commit,
    )

    split_summary_rows = []
    for dataset in split.as_list():
        subset = labelled_all[
            labelled_all["split_name"] == dataset.name
        ]
        split_summary_rows.append(
            {
                "split_name": dataset.name,
                "start_date": dataset.start_date.isoformat(),
                "end_date": dataset.end_date.isoformat(),
                "trade_count": int(len(subset)),
                "first_trade_date": (
                    subset["entry_date"].min().date().isoformat()
                    if not subset.empty
                    else None
                ),
                "last_trade_date": (
                    subset["entry_date"].max().date().isoformat()
                    if not subset.empty
                    else None
                ),
            }
        )
    pd.DataFrame(split_summary_rows).to_csv(
        output_path / "split_summary.csv",
        index=False,
    )

    trade_log = labelled_all.copy()
    trade_log["exit_crosses_split"] = trade_log.index.isin(
        pd.concat(
            [train_crossing, validation_crossing],
            ignore_index=True,
        ).index
    )
    trade_log.to_csv(output_path / "trade_log.csv", index=False)
    pd.concat(
        [train_crossing, validation_crossing],
        ignore_index=True,
    ).to_csv(
        output_path / "boundary_crossing_trades.csv",
        index=False,
    )

    with (output_path / "selected_parameters.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {"minimum_volatility_ratio": selected_threshold},
            file,
            ensure_ascii=False,
            indent=2,
        )
    for name, metrics in [
        ("train", best_train_metrics),
        ("validation", best_validation_metrics),
        ("test", test_metrics),
    ]:
        with (output_path / f"{name}_metrics.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)

    return {
        "selected_threshold": selected_threshold,
        "test_metrics": test_metrics,
        "output_directory": str(output_path),
    }
