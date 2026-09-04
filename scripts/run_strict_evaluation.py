"""End-to-end strict out-of-sample evaluation on real SPY data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
)
from src.config import load_config
from src.evaluation.configuration import build_fixed_split
from src.evaluation.runner import run_strict_evaluation
from src.market_data.underlying_data import load_price_data
from src.run_context import initialise_run


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict train/validation/test evaluation."
    )
    parser.add_argument(
        "--allow-test-repeat",
        action="store_true",
        help="Allow re-evaluation of the locked test set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_config("configs/default.yaml")
    context = initialise_run(
        config=config,
        config_path="configs/default.yaml",
        command="python scripts/run_strict_evaluation.py",
    )
    evaluation = config.get("evaluation", {})
    split = build_fixed_split(config)
    price_data = load_price_data(
        Path("data/raw/spy_daily_adjusted.csv")
    )

    result = run_strict_evaluation(
        price_data=price_data,
        candidate_thresholds=[1.00, 1.05, 1.10, 1.15, 1.20],
        split=split,
        output_directory=context.output_directory / "evaluation",
        run_id=context.run_id,
        trade_config=LongStraddleBacktestConfig(
            days_to_expiry=30,
            delta_threshold=5.0,
        ),
        rolling_config=RollingBacktestConfig(
            entry_spacing_trading_days=30,
            initial_capital=100_000.0,
        ),
        minimum_validation_trades=evaluation.get(
            "minimum_validation_trades",
            6,
        ),
        lag_regime_signal=evaluation.get(
            "lag_regime_signal",
            True,
        ),
        test_lock_path="evaluation/test_evaluation_log.json",
        allow_test_repeat=args.allow_test_repeat,
        git_commit=context.metadata["reproducibility"]["git_commit"],
    )

    print(f"run_id: {context.run_id}")
    print(
        "selected threshold: "
        f"{result['selected_threshold']:.2f}"
    )
    print(
        "test metrics: "
        f"trades={result['test_metrics']['trade_count']}, "
        "annualized_sharpe="
        f"{result['test_metrics']['annualized_sharpe_estimate']:.3f}, "
        f"total_pnl={result['test_metrics']['total_pnl']:.0f}"
    )
    print(f"output: {result['output_directory']}")


if __name__ == "__main__":
    main()
