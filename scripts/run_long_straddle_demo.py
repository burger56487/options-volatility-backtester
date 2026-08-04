from pathlib import Path

import pandas as pd

from src.market_data.synthetic_option_chain import (
    select_atm_straddle,
)
from src.strategy.long_straddle import (
    build_long_atm_straddle,
)


def main() -> None:
    chain_path = Path(
        "data/processed/spy_synthetic_option_chain_latest.csv"
    )

    chain = pd.read_csv(
        chain_path,
        parse_dates=[
            "valuation_date",
            "expiry_date",
        ],
    )

    straddle_chain = select_atm_straddle(
        chain=chain,
        days_to_expiry=30,
    )

    straddle = build_long_atm_straddle(
        chain=straddle_chain,
        quantity=1,
        multiplier=100,
    )

    valuation_date = pd.Timestamp(
        straddle_chain["valuation_date"].iloc[0]
    )
    spot = float(straddle_chain["spot"].iloc[0])
    call_volatility = float(
        straddle_chain.loc[
            straddle_chain["option_type"] == "call",
            "implied_volatility",
        ].iloc[0]
    )
    put_volatility = float(
        straddle_chain.loc[
            straddle_chain["option_type"] == "put",
            "implied_volatility",
        ].iloc[0]
    )

    risk_free_rate = 0.04
    dividend_yield = 0.012

    greeks = straddle.combined_greeks(
        valuation_date=valuation_date,
        spot=spot,
        risk_free_rate=risk_free_rate,
        call_volatility=call_volatility,
        put_volatility=put_volatility,
        dividend_yield=dividend_yield,
    )

    market_value = straddle.market_value(
        valuation_date=valuation_date,
        spot=spot,
        risk_free_rate=risk_free_rate,
        call_volatility=call_volatility,
        put_volatility=put_volatility,
        dividend_yield=dividend_yield,
    )

    print("=" * 64)
    print("LONG ATM STRADDLE CONSTRUCTION DEMO")
    print("=" * 64)
    print()

    print(f"Valuation date: {valuation_date.date()}")
    print(f"Expiry date: {straddle.expiry_date.date()}")
    print(f"Spot: {spot:.4f}")
    print(f"Strike: {straddle.strike:.2f}")
    print(f"Contracts: {straddle.quantity}")
    print()

    print("Entry")
    print("-" * 64)
    print(
        "Call entry price (ask): "
        f"{straddle.call_position.entry_price:.4f}"
    )
    print(
        "Put entry price (ask):  "
        f"{straddle.put_position.entry_price:.4f}"
    )
    print(f"Total entry cost:       {straddle.entry_cost:.2f}")
    print()

    print("Current valuation")
    print("-" * 64)
    print(f"Market value: {market_value:.2f}")
    print(
        f"Option P&L:   "
        f"{market_value - straddle.entry_cost:.2f}"
    )
    print()

    print("Combined Greeks")
    print("-" * 64)

    for name, value in greeks.items():
        print(f"{name.capitalize():<8}: {value:.6f}")


if __name__ == "__main__":
    main()
