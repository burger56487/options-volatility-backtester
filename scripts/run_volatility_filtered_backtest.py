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
from src.backtest.volatility_filter import (
    VolatilityRegimeFilter,
)
from src.market_data.underlying_data import load_price_data


def print_summary(
    title: str,
    summary: dict[str, float | int | str],
) -> None:
    print(title)
    print("-" * 72)

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print()


def main() -> None:
    input_path = Path("data/raw/spy_daily_adjusted.csv")
    output_directory = Path(
        "outputs/volatility_filtered_backtest"
    )
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

    regime_filter = VolatilityRegimeFilter(
        short_window=20,
        long_window=252,
        minimum_volatility_ratio=1.10,
    )

    print("=" * 72)
    print("VOLATILITY-REGIME FILTERED LONG STRADDLE BACKTEST")
    print("=" * 72)
    print()

    baseline_result = run_rolling_long_straddle_backtest(
        price_data=data,
        trade_config=trade_config,
        rolling_config=rolling_config,
    )

    filtered_result = run_rolling_long_straddle_backtest(
        price_data=data,
        trade_config=trade_config,
        rolling_config=rolling_config,
        regime_filter=regime_filter,
    )

    save_rolling_backtest_result(
        result=baseline_result,
        output_directory=output_directory,
        prefix="baseline",
    )

    save_rolling_backtest_result(
        result=filtered_result,
        output_directory=output_directory,
        prefix="volatility_filtered",
    )

    print_summary(
        "Baseline result",
        baseline_result.summary,
    )

    print_summary(
        "Volatility-filtered result",
        filtered_result.summary,
    )

    comparison = pd.DataFrame(
        [
            {
                "strategy": "Baseline",
                **baseline_result.summary,
            },
            {
                "strategy": "Volatility Filtered",
                **filtered_result.summary,
            },
        ]
    )

    comparison_path = (
        output_directory / "baseline_vs_filtered_comparison.csv"
    )
    comparison.to_csv(comparison_path, index=False)

    figure_directory.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.plot(
        baseline_result.equity_curve.index,
        baseline_result.equity_curve["equity"],
        marker="o",
        label="Baseline",
    )

    plt.plot(
        filtered_result.equity_curve.index,
        filtered_result.equity_curve["equity"],
        marker="o",
        label="Volatility Filtered",
    )

    plt.title("Baseline vs Volatility-Filtered Long Straddle")
    plt.suptitle("Synthetic option quotes | simulated execution", fontsize=9)
    plt.xlabel("Trade Entry Date")
    plt.ylabel("Illustrative Capital")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    figure_path = (
        figure_directory
        / "spy_baseline_vs_volatility_filtered.png"
    )

    plt.savefig(figure_path, dpi=160)
    plt.close()

    print(f"Saved results to: {output_directory}")
    print(f"Saved comparison figure to: {figure_path}")


if __name__ == "__main__":
    main()
