"""Demo of the upgraded statistical-validation and P&L-attribution modules."""

from pathlib import Path

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
    run_long_straddle_backtest,
)
from src.backtest.pnl_attribution import (
    attribute_daily_pnl,
    attribution_summary,
)
from src.backtest.rolling_backtest import (
    RollingBacktestConfig,
    run_rolling_long_straddle_backtest,
)
from src.market_data.underlying_data import load_price_data


def main() -> None:
    data = load_price_data(
        Path("data/raw/spy_daily_adjusted.csv")
    )

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

    rolling = run_rolling_long_straddle_backtest(
        price_data=data,
        trade_config=trade_config,
        rolling_config=RollingBacktestConfig(
            entry_spacing_trading_days=30,
            initial_capital=100_000.0,
            confidence_level=0.95,
        ),
    )

    metric_keys = [
        "number_of_trades",
        "mean_trade_return",
        "sharpe_like_ratio",
        "mean_trade_return_ci_low",
        "mean_trade_return_ci_high",
        "sharpe_like_ratio_ci_low",
        "sharpe_like_ratio_ci_high",
        "annualized_sharpe_estimate",
        "annualized_sharpe_ci_low",
        "annualized_sharpe_ci_high",
        "trades_per_year_assumed",
        "metric_definitions",
    ]

    print("ROLLING BACKTEST METRICS (with CI and explicit definitions)")
    for key in metric_keys:
        value = rolling.summary[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    first_entry = rolling.trade_results["entry_date"].iloc[0]
    single = run_long_straddle_backtest(
        price_data=data,
        entry_date=first_entry,
        config=trade_config,
    )

    attribution = attribute_daily_pnl(single.equity_curve)
    summary = attribution_summary(attribution)

    print()
    print("SINGLE-TRADE P&L ATTRIBUTION (first rolling trade)")
    print(
        "entry_date: "
        f"{single.summary['entry_date']}  "
        f"final_pnl: {single.summary['final_pnl']:.2f}"
    )
    for key in [
        "delta_total",
        "gamma_total",
        "vega_total",
        "theta_total",
        "rho_total",
        "cost_total",
        "actual_pnl_total",
        "residual_total",
        "abs_residual_ratio",
    ]:
        print(f"{key}: {summary[key]:.4f}")


if __name__ == "__main__":
    main()
