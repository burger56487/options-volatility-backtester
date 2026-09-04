"""Out-of-sample evaluation of the volatility-regime filter threshold."""

from __future__ import annotations

from pathlib import Path

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.oos_volatility_filter import (
    evaluate_filter_out_of_sample,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
)
from src.market_data.underlying_data import load_price_data


def main() -> None:
    data = load_price_data(
        Path("data/raw/spy_daily_adjusted.csv")
    )

    result = evaluate_filter_out_of_sample(
        price_data=data,
        candidate_thresholds=[1.00, 1.05, 1.10, 1.15, 1.20],
        trade_config=LongStraddleBacktestConfig(
            days_to_expiry=30,
            delta_threshold=5.0,
        ),
        rolling_config=RollingBacktestConfig(
            entry_spacing_trading_days=30,
            initial_capital=100_000.0,
        ),
        train_fraction=0.6,
    )

    print("threshold selection scores (training window only):")
    print("threshold | annualized_sharpe | trades")
    for threshold, score, trades in result.threshold_scores:
        print(f"{threshold:.2f} | {score:9.3f} | {trades}")

    print()
    print(
        "selected threshold: "
        f"{result.selected_threshold:.2f}"
    )

    for label, backtest in [
        ("train (filtered)", result.train_result),
        ("test  (filtered)", result.test_result),
        ("test  (baseline) ", result.baseline_test_result),
    ]:
        summary = backtest.summary
        print(
            f"{label}: trades={summary['number_of_trades']}, "
            f"total_pnl={summary['total_pnl']:.0f}, "
            "annualized_sharpe="
            f"{summary['annualized_sharpe_estimate']:.3f}"
        )


if __name__ == "__main__":
    main()
