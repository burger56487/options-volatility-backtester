from pathlib import Path

import json
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
    run_long_straddle_backtest,
    save_backtest_result,
)
from src.market_data.underlying_data import load_price_data


def main() -> None:
    input_path = Path("data/raw/spy_daily_adjusted.csv")
    output_directory = Path("outputs/single_trade")

    data = load_price_data(input_path)

    # Use a date with enough prior history and enough subsequent observations.
    entry_date = pd.Timestamp("2025-03-03")

    config = LongStraddleBacktestConfig(
        days_to_expiry=30,
        quantity=1,
        multiplier=100,
        risk_free_rate=0.04,
        dividend_yield=0.012,
        delta_threshold=1.0,
        allow_fractional_shares=False,
        commission_per_share=0.005,
        underlying_slippage_bps=1.0,
    )

    print("=" * 72)
    print("SINGLE-TRADE DELTA-HEDGED LONG STRADDLE BACKTEST")
    print("=" * 72)
    print()

    result = run_long_straddle_backtest(
        price_data=data,
        entry_date=entry_date,
        config=config,
    )

    save_backtest_result(
        result=result,
        output_directory=output_directory,
        prefix="spy_long_straddle",
    )

    print("Backtest summary")
    print("-" * 72)

    for key, value in result.summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print()
    print("Last five equity-curve observations")
    print("-" * 72)
    print(
        result.equity_curve.tail()[
            [
                "spot",
                "option_value",
                "option_delta",
                "hedge_position",
                "post_hedge_delta",
                "cumulative_turnover",
                "cumulative_hedge_costs",
                "total_pnl",
            ]
        ]
        .round(4)
        .to_string()
    )

    figure_directory = Path("outputs/figures")
    figure_directory.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(
        result.equity_curve.index,
        result.equity_curve["total_pnl"],
        label="Delta-Hedged Long Straddle P&L",
        linewidth=1.5,
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("SPY Delta-Hedged Long Straddle P&L")
    plt.xlabel("Date")
    plt.ylabel("P&L")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figure_directory / "spy_single_trade_equity_curve.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(
        result.equity_curve.index,
        result.equity_curve["option_delta"],
        label="Option Delta",
        linewidth=1.2,
    )
    plt.plot(
        result.equity_curve.index,
        result.equity_curve["post_hedge_delta"],
        label="Post-Hedge Delta",
        linewidth=1.2,
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Option Delta Before and After Underlying Hedge")
    plt.xlabel("Date")
    plt.ylabel("Delta Exposure")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figure_directory / "spy_single_trade_delta_exposure.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.step(
        result.equity_curve.index,
        result.equity_curve["hedge_position"],
        where="post",
        label="SPY Hedge Position",
        linewidth=1.2,
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Dynamic SPY Delta-Hedge Position")
    plt.xlabel("Date")
    plt.ylabel("Shares")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        figure_directory / "spy_single_trade_hedge_position.png",
        dpi=160,
    )
    plt.close()

    print()
    print(
        "Saved artifacts to: "
        f"{output_directory}"
    )
    print(
        "Saved figures to: "
        f"{figure_directory}"
    )


if __name__ == "__main__":
    main()
