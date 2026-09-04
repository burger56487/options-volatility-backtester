"""One-shot rolling account-engine straddle demo with saved outputs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.account_delta_hedge import (
    AccountHedgeConfig,
    run_rolling_account_straddle_backtest,
)
from src.market_data.underlying_data import load_price_data


def main() -> None:
    data = load_price_data(
        Path("data/raw/spy_daily_adjusted.csv")
    )
    result = run_rolling_account_straddle_backtest(
        price_data=data,
        config=AccountHedgeConfig(
            expiry_days=30,
            volatility=0.20,
        ),
        entry_spacing=60,
        minimum_history=252,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path("outputs") / f"account_rolling_{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    result["trade_results"].to_csv(
        output / "trade_results.csv",
        index=False,
    )
    result["equity_curve"].to_csv(
        output / "equity_curve.csv",
    )
    with (output / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result["summary"],
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"trades: {result['summary']['number_of_trades']}")
    print(
        "total pnl: "
        f"{result['summary']['total_pnl']:.2f}"
    )
    print(
        "reconciliation failures: "
        f"{result['summary']['reconciliation_failures']}"
    )
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
