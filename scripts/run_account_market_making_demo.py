"""Demo: account-engine market making with Greeks-aware quotes and hedging.

Option quotes are synthetic BSM fair values on a deterministic random-walk
spot path; the run exercises the full account/execution/risk pipeline and
writes snapshots, fills and a summary.  This is a research simulation, not
live trading.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.account_market_making import (
    AccountMarketMakingConfig,
    run_account_market_making,
    save_account_market_making_result,
)


def main() -> None:
    rng = np.random.default_rng(9)
    periods = 45
    dates = pd.date_range("2026-01-02", periods=periods, freq="B")
    returns = rng.normal(0.0, 0.0025, periods)
    prices = 100.0 * np.cumprod(1.0 + returns)
    data = pd.DataFrame({"Close": prices}, index=dates)
    config = AccountMarketMakingConfig(
        strikes=(0.95, 1.00, 1.05),
        expiry_days=(10, 20, 30),
        include_puts=True,
        arrival_probability=0.6,
        volatility=0.25,
        seed=11,
    )
    result = run_account_market_making(
        price_data=data,
        config=config,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = save_account_market_making_result(
        result,
        Path("outputs") / f"account_market_making_{stamp}",
    )
    print(f"fills: {result['fill_count']}")
    print(f"rejected fills: {result['rejected_fill_count']}")
    print(f"halted: {result['halted']}")
    print(f"final equity: {result['final_equity']:.2f}")
    print(f"total pnl: {result['total_pnl']:.2f}")
    print(
        "reconciliation passed: "
        f"{result['reconciliation_passed']}"
    )
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
