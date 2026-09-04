"""Recompute the legacy rolling backtest with the new metric framework."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
    run_rolling_long_straddle_backtest,
)
from src.evaluation.report_builder import save_markdown_report
from src.market_data.underlying_data import load_price_data
from src.performance.reporting import compute_performance_report


def main() -> None:
    data = load_price_data(
        Path("data/raw/spy_daily_adjusted.csv")
    )
    result = run_rolling_long_straddle_backtest(
        price_data=data,
        trade_config=LongStraddleBacktestConfig(
            days_to_expiry=30,
            delta_threshold=5.0,
        ),
        rolling_config=RollingBacktestConfig(
            entry_spacing_trading_days=30,
            initial_capital=100_000.0,
        ),
    )
    performance = compute_performance_report(
        result.equity_curve["equity"],
        evaluation_mode=str(
            result.summary.get(
                "evaluation_mode",
                "legacy_rolling",
            )
        ),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("outputs") / f"recomputed_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "performance_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "legacy_summary": {
                    key: result.summary[key]
                    for key in (
                        "number_of_trades",
                        "total_pnl",
                        "legacy_sharpe_like_ratio",
                        "annualized_sharpe_estimate",
                        "option_data_type",
                        "execution_type",
                    )
                    if key in result.summary
                },
                "performance": performance,
                "note": (
                    "Legacy close-to-close rolling backtest recomputed "
                    "with daily-frequency metrics; account/execution "
                    "engine not yet wired into this path."
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    save_markdown_report(
        output_path=output_dir / "report.md",
        title="Legacy Rolling Long Straddle - Recomputed Metrics",
        sections={
            "Performance": performance,
            "Legacy summary": dict(result.summary),
        },
    )
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
