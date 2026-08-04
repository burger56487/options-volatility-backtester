from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.regime_sensitivity import (
    run_regime_threshold_sensitivity,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
)
from src.market_data.underlying_data import load_price_data


def main() -> None:
    input_path = Path("data/raw/spy_daily_adjusted.csv")
    output_directory = Path("outputs/regime_sensitivity")
    figure_directory = Path("outputs/figures")

    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

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

    thresholds = (
        0.90,
        1.00,
        1.05,
        1.10,
        1.20,
        1.30,
    )

    print("=" * 72)
    print("VOLATILITY-REGIME THRESHOLD SENSITIVITY")
    print("=" * 72)
    print()
    print(f"Thresholds: {thresholds}")
    print()

    results = run_regime_threshold_sensitivity(
        price_data=data,
        trade_config=trade_config,
        rolling_config=rolling_config,
        thresholds=thresholds,
        short_window=20,
        long_window=252,
    )

    output_path = (
        output_directory
        / "volatility_regime_threshold_sensitivity.csv"
    )
    results.to_csv(output_path, index=False)

    print("Results")
    print("-" * 72)

    display_columns = [
        "volatility_ratio_threshold",
        "number_of_trades",
        "total_pnl",
        "total_return_on_initial_capital",
        "win_rate",
        "portfolio_max_drawdown_pct",
        "sharpe_like_ratio",
        "var",
        "expected_shortfall",
        "total_hedge_costs",
    ]

    display_results = results[display_columns].copy()

    numeric_columns = [
        "volatility_ratio_threshold",
        "total_pnl",
        "total_return_on_initial_capital",
        "win_rate",
        "portfolio_max_drawdown_pct",
        "sharpe_like_ratio",
        "var",
        "expected_shortfall",
        "total_hedge_costs",
    ]

    display_results[numeric_columns] = (
        display_results[numeric_columns].round(4)
    )

    print(display_results.to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.plot(
        results["volatility_ratio_threshold"],
        results["total_pnl"],
        marker="o",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title(
        "Long Straddle P&L by Volatility-Regime Threshold"
    )
    plt.xlabel("Short / Long Realised Volatility Threshold")
    plt.ylabel("Total P&L")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        figure_directory
        / "spy_regime_threshold_sensitivity_pnl.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        results["volatility_ratio_threshold"],
        results["number_of_trades"],
        marker="o",
        color="tab:orange",
    )
    plt.title(
        "Trade Count by Volatility-Regime Threshold"
    )
    plt.xlabel("Short / Long Realised Volatility Threshold")
    plt.ylabel("Number of Trades")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        figure_directory
        / "spy_regime_threshold_sensitivity_trade_count.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        results["volatility_ratio_threshold"],
        results["portfolio_max_drawdown_pct"] * 100.0,
        marker="o",
        color="tab:red",
    )
    plt.title(
        "Portfolio Drawdown by Volatility-Regime Threshold"
    )
    plt.xlabel("Short / Long Realised Volatility Threshold")
    plt.ylabel("Maximum Drawdown (%)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        figure_directory
        / "spy_regime_threshold_sensitivity_drawdown.png",
        dpi=160,
    )
    plt.close()

    print()
    print(f"Saved results to: {output_path}")
    print(f"Saved figures to: {figure_directory}")


if __name__ == "__main__":
    main()
