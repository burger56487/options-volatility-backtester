from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
    run_rolling_long_straddle_backtest,
    save_rolling_backtest_result,
)
from src.market_data.underlying_data import load_price_data


def main() -> None:
    input_path = Path("data/raw/spy_daily_adjusted.csv")
    output_directory = Path("outputs/rolling_backtest")
    figure_directory = Path("outputs/figures")

    data = load_price_data(input_path)

    trade_config = LongStraddleBacktestConfig(
        days_to_expiry=30,
        quantity=1,
        multiplier=100,
        risk_free_rate=0.04,
        dividend_yield=0.012,
        delta_threshold=5.0,
        allow_fractional_shares=False,
        commission_per_share=0.005,
        underlying_slippage_bps=1.0,
    )

    rolling_config = RollingBacktestConfig(
        entry_spacing_trading_days=30,
        initial_capital=100_000.0,
        confidence_level=0.95,
    )

    print("=" * 72)
    print("ROLLING DELTA-HEDGED LONG STRADDLE BACKTEST")
    print("=" * 72)
    print()

    result = run_rolling_long_straddle_backtest(
        price_data=data,
        trade_config=trade_config,
        rolling_config=rolling_config,
    )

    save_rolling_backtest_result(
        result=result,
        output_directory=output_directory,
        prefix="spy_rolling_long_straddle",
    )

    print("Performance summary")
    print("-" * 72)

    for key, value in result.summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print()
    print("Trade results")
    print("-" * 72)
    print(
        result.trade_results[
            [
                "entry_date",
                "expiry_date",
                "entry_spot",
                "final_spot",
                "final_pnl",
                "trade_return",
                "max_drawdown",
                "number_of_hedge_trades",
                "cumulative_hedge_costs",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    figure_directory.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(
        result.equity_curve.index,
        result.equity_curve["equity"],
        marker="o",
        linewidth=1.5,
        label="Illustrative Equity",
    )
    plt.title("Rolling Delta-Hedged Long Straddle Equity Curve")
    plt.xlabel("Trade Entry Date")
    plt.ylabel("Capital")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figure_directory
        / "spy_rolling_long_straddle_equity_curve.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(
        result.trade_results["entry_date"].astype(str),
        result.trade_results["final_pnl"],
        color=[
            "tab:green"
            if value >= 0
            else "tab:red"
            for value in result.trade_results["final_pnl"]
        ],
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Rolling Long Straddle P&L by Trade")
    plt.xlabel("Entry Date")
    plt.ylabel("Final P&L")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        figure_directory
        / "spy_rolling_long_straddle_trade_pnl.png",
        dpi=160,
    )
    plt.close()

    print()
    print(f"Saved output to: {output_directory}")
    print(f"Saved figures to: {figure_directory}")


if __name__ == "__main__":
    main()
