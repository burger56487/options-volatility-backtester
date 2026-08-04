from pathlib import Path

import pandas as pd

from src.market_data.synthetic_option_chain import (
    select_atm_straddle,
)
from src.strategy.delta_hedging import (
    DeltaHedger,
    UnderlyingTransactionCostModel,
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

    option_greeks = straddle.combined_greeks(
        valuation_date=valuation_date,
        spot=spot,
        risk_free_rate=risk_free_rate,
        call_volatility=call_volatility,
        put_volatility=put_volatility,
        dividend_yield=dividend_yield,
    )

    hedger = DeltaHedger(
        cost_model=UnderlyingTransactionCostModel(
            commission_per_share=0.005,
            slippage_bps=1.0,
        ),
        delta_threshold=1.0,
        allow_fractional_shares=False,
    )

    trade = hedger.rebalance(
        trade_date=valuation_date,
        spot=spot,
        option_delta=option_greeks["delta"],
    )

    print("=" * 64)
    print("DELTA HEDGING DEMO")
    print("=" * 64)
    print()

    print(f"Valuation date: {valuation_date.date()}")
    print(f"SPY spot: {spot:.4f}")
    print(f"Option portfolio delta: {option_greeks['delta']:.6f}")
    print()

    if trade is None:
        print("No hedge trade executed: delta adjustment below threshold.")
        return

    print("Hedge trade")
    print("-" * 64)
    print(f"Trade quantity: {trade.quantity:.0f} SPY shares")
    print(f"Reference spot: {trade.reference_spot:.4f}")
    print(f"Execution price: {trade.execution_price:.4f}")
    print(f"Transaction cost: {trade.transaction_cost:.4f}")
    print(f"Position before trade: {trade.pre_trade_position:.0f}")
    print(f"Position after trade: {trade.post_trade_position:.0f}")
    print(f"Delta after hedge: {trade.post_hedge_delta:.6f}")
    print()

    print("Hedge account")
    print("-" * 64)
    print(f"Underlying position: {hedger.position:.0f}")
    print(f"Cash balance: {hedger.cash:.4f}")
    print(f"Market value: {hedger.market_value(spot):.4f}")
    print(f"Total equity: {hedger.total_equity(spot):.4f}")
    print(
        "Cumulative turnover: "
        f"{hedger.cumulative_turnover:.4f}"
    )
    print(
        "Cumulative transaction costs: "
        f"{hedger.cumulative_transaction_costs:.4f}"
    )


if __name__ == "__main__":
    main()
