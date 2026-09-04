"""Demo: account-engine-driven delta hedge on real SPY history."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.account_delta_hedge import (
    AccountHedgeConfig,
    run_account_delta_hedge,
    save_account_hedge_result,
)
from src.market_data.underlying_data import load_price_data


def main() -> None:
    data = load_price_data(
        Path("data/raw/spy_daily_adjusted.csv")
    )
    entry = data.index[300]
    result = run_account_delta_hedge(
        price_data=data,
        entry_date=entry,
        config=AccountHedgeConfig(expiry_days=30),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = save_account_hedge_result(
        result,
        Path("outputs") / f"account_hedge_{stamp}",
    )
    print(f"entry: {entry.date()}")
    print(f"final equity: {result['final_equity']:.2f}")
    print(f"total pnl: {result['total_pnl']:.2f}")
    print(
        "reconciliation passed: "
        f"{result['reconciliation_passed']}"
    )
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
