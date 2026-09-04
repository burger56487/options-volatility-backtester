"""End-to-end rolling backtest run with full run metadata."""

from __future__ import annotations

import argparse
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
from src.config import load_config
from src.manifest import create_manifest
from src.market_data.pipeline import (
    run_market_data_pipeline,
    underlying_clean_to_price_frame,
)
from src.plotting import build_research_subtitle
from src.research_guardrails import validate_research_claims
from src.run_context import initialise_run


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the rolling delta-hedged long-straddle backtest "
        "with traceable run metadata."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="YAML research configuration path.",
    )
    parser.add_argument(
        "--price-data",
        default="data/raw/spy_daily_adjusted.csv",
        help="CSV path of the underlying daily price data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    validate_research_claims(
        config=config,
        run_label="波动率过滤策略研究（合成期权报价）",
    )

    context = initialise_run(
        config=config,
        config_path=args.config,
        command=" ".join(sys.argv),
    )

    legacy = pd.read_csv(args.price_data)
    legacy["Date"] = pd.to_datetime(legacy["Date"]).dt.date
    unified_path = Path("data/raw/underlying/underlying.csv")
    unified_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": legacy["Date"],
            "symbol": "SPY",
            "open": legacy["Open"],
            "high": legacy["High"],
            "low": legacy["Low"],
            "close": legacy["Close"],
            "adjusted_close": legacy["Close"],
            "volume": legacy["Volume"],
        }
    ).to_csv(unified_path, index=False)
    data_quality = config.get("data_quality", {})
    market_data_dir = context.output_directory / "market_data"
    run_market_data_pipeline(
        underlying_input_path=unified_path,
        option_input_path=Path(
            "data/sample/option_quotes_sample.csv"
        ),
        output_directory=market_data_dir,
        run_id=context.run_id,
        underlying_source=config["data"]["underlying"]["source"],
        fail_on_invalid=data_quality.get("fail_on_invalid", True),
    )
    data = underlying_clean_to_price_frame(
        market_data_dir / "underlying_clean.csv"
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
    rolling_config = RollingBacktestConfig(
        entry_spacing_trading_days=30,
        initial_capital=100_000.0,
        confidence_level=0.95,
    )

    result = run_rolling_long_straddle_backtest(
        price_data=data,
        trade_config=trade_config,
        rolling_config=rolling_config,
    )

    output = context.output_directory
    trades = result.trade_results.copy()
    trades["run_id"] = context.run_id
    trades["option_data_type"] = config["data"]["options"]["data_type"]
    trades["execution_mode"] = config["execution"]["mode"]
    trades["evaluation_mode"] = config["research"]["evaluation_mode"]
    trades.to_csv(output / "backtest_results.csv", index=False)

    equity = result.equity_curve.copy()
    equity["run_id"] = context.run_id
    equity.to_csv(output / "equity_curve.csv")

    summary = dict(result.summary)
    summary["run_id"] = context.run_id
    pd.DataFrame([summary]).to_json(
        output / "summary.json",
        orient="records",
        indent=2,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure_directory = output / "figures"
        figure_directory.mkdir(exist_ok=True)
        fig, axis = plt.subplots(figsize=(12, 6))
        axis.plot(
            equity.index,
            equity["equity"],
            marker="o",
            linewidth=1.5,
        )
        axis.set_title(
            build_research_subtitle(config),
            fontsize=9,
        )
        axis.set_xlabel("Trade entry date")
        axis.set_ylabel("Illustrative equity")
        axis.grid(alpha=0.25)
        fig.suptitle(
            "Rolling delta-hedged long straddle "
            "(synthetic option quotes)"
        )
        fig.text(
            0.5,
            0.01,
            "Research simulation; synthetic option quotes do not "
            "represent real historical option-market performance.",
            ha="center",
            fontsize=8,
            color="dimgray",
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        fig.savefig(
            figure_directory / "equity_curve.png",
            dpi=160,
        )
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - figures are best effort
        print(f"figure skipped: {exc}")

    manifest_path = create_manifest(output)

    print(f"run_id: {context.run_id}")
    print(f"output_directory: {output}")
    print(
        "trades: "
        f"{summary['number_of_trades']}, "
        f"total_pnl: {summary['total_pnl']:.2f}"
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
