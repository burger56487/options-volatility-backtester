"""Train/validation/test strategy survey with paper PnL."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.backtest.timeline import three_way_split
from src.market_data.underlying_data import load_price_data
from src.strategy_lib.evaluation import paper_strategy_pnl
from src.strategy_lib.strategies import long_straddle, long_strangle
from src.strategy_lib.strategies2 import calendar_spread, risk_reversal


def main() -> None:
    data = load_price_data(
        Path("data/raw/spy_daily_adjusted.csv")
    )
    train, validation, test = three_way_split(
        data,
        fractions=(0.5, 0.25, 0.25),
    )
    strategies = {
        "long_straddle": long_straddle(30),
        "long_strangle": long_strangle(30),
        "calendar_spread": calendar_spread(30, 60),
        "risk_reversal": risk_reversal(30),
    }
    horizon = 10
    rows = []
    for split_name, window in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if len(window) <= horizon:
            continue
        entry = float(window["Close"].iloc[0])
        exit_spot = float(window["Close"].iloc[horizon])
        for name, definition in strategies.items():
            pnl = paper_strategy_pnl(
                definition,
                entry_spot=entry,
                exit_spot=exit_spot,
                days_held=horizon,
                risk_free_rate=0.04,
                volatility=0.2,
            )
            rows.append(
                {
                    "split": split_name,
                    "strategy": name,
                    "entry_spot": entry,
                    "exit_spot": exit_spot,
                    "paper_pnl": pnl,
                }
            )
    output = Path("outputs") / "strategy_survey_oos.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
