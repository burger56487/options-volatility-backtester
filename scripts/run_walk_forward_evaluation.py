"""Expanding-window walk-forward evaluation over SPY history."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
    run_rolling_long_straddle_backtest,
)
from src.evaluation.walk_forward import (
    generate_expanding_window_folds,
)
from src.market_data.underlying_data import load_price_data


def main() -> None:
    data = load_price_data(
        Path("data/raw/spy_daily_adjusted.csv")
    )
    folds = generate_expanding_window_folds(
        dates=data.index.to_series(),
        minimum_train_observations=504,
        validation_observations=126,
        test_observations=126,
        step_observations=126,
    )
    rows = []
    for fold in folds[:6]:
        window = data.loc[fold.test_start : fold.test_end]
        result = run_rolling_long_straddle_backtest(
            price_data=window,
            trade_config=LongStraddleBacktestConfig(
                days_to_expiry=30,
                delta_threshold=5.0,
            ),
            rolling_config=RollingBacktestConfig(
                entry_spacing_trading_days=30,
            ),
        )
        rows.append(
            {
                "fold_id": fold.fold_id,
                "test_start": fold.test_start.date(),
                "test_end": fold.test_end.date(),
                "trades": result.summary["number_of_trades"],
                "total_pnl": result.summary["total_pnl"],
                "annualized_sharpe": result.summary[
                    "annualized_sharpe_estimate"
                ],
            }
        )
    output = Path("outputs") / "walk_forward_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
