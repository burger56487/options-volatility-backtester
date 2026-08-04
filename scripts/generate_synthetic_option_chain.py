from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.market_data.realized_volatility import (
    add_realized_volatility_features,
)
from src.market_data.synthetic_option_chain import (
    blended_realised_volatility,
    create_synthetic_option_chain,
)
from src.market_data.underlying_data import (
    load_price_data,
)


def main() -> None:
    input_path = Path("data/raw/spy_daily_adjusted.csv")
    output_path = Path(
        "data/processed/spy_synthetic_option_chain_latest.csv"
    )
    smile_figure_path = Path(
        "outputs/figures/spy_synthetic_volatility_smile.png"
    )
    term_figure_path = Path(
        "outputs/figures/spy_synthetic_term_structure.png"
    )

    risk_free_rate = 0.04
    dividend_yield = 0.012

    print("=" * 64)
    print("SYNTHETIC SPY OPTION CHAIN GENERATION")
    print("=" * 64)
    print()

    data = load_price_data(input_path)

    featured_data = add_realized_volatility_features(
        data=data,
        windows=(20, 60, 252),
    ).dropna(
        subset=[
            "realised_vol_20d",
            "realised_vol_60d",
            "realised_vol_252d",
        ]
    )

    latest = featured_data.iloc[-1]

    base_volatility = blended_realised_volatility(
        realized_vol_20d=float(latest["realised_vol_20d"]),
        realized_vol_60d=float(latest["realised_vol_60d"]),
        realized_vol_252d=float(latest["realised_vol_252d"]),
    )

    chain = create_synthetic_option_chain(
        valuation_date=pd.Timestamp(latest.name),
        spot=float(latest["Close"]),
        base_volatility=base_volatility,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chain.to_csv(output_path, index=False)

    print(f"Valuation date: {latest.name.date()}")
    print(f"SPY spot price: {latest['Close']:.4f}")
    print(f"Blended realised volatility: {base_volatility:.2%}")
    print(f"Generated option contracts: {len(chain)}")
    print(f"Saved option chain to: {output_path}")
    print()

    smile_figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for days in sorted(chain["days_to_expiry"].unique()):
        subset = chain.loc[
            (chain["days_to_expiry"] == days)
            & (chain["option_type"] == "put")
        ]

        plt.plot(
            subset["moneyness"],
            subset["implied_volatility"],
            marker="o",
            label=f"{days} days",
        )

    plt.title("Synthetic SPY Put Implied-Volatility Smile")
    plt.xlabel("Strike / Spot")
    plt.ylabel("Implied Volatility")
    plt.legend(title="Expiry")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(smile_figure_path, dpi=160)
    plt.close()

    call_chain = chain.loc[
        chain["option_type"] == "call"
    ].copy()

    call_chain["atm_distance"] = (
        call_chain["moneyness"] - 1.0
    ).abs()

    atm_indices = call_chain.groupby(
        "days_to_expiry"
    )["atm_distance"].idxmin()

    atm_chain = call_chain.loc[
        atm_indices
    ].sort_values("days_to_expiry")


    plt.figure(figsize=(10, 6))
    plt.plot(
        atm_chain["days_to_expiry"],
        atm_chain["implied_volatility"],
        marker="o",
        linewidth=1.5,
    )
    plt.title("Synthetic SPY ATM Implied-Volatility Term Structure")
    plt.xlabel("Days to Expiry")
    plt.ylabel("Implied Volatility")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(term_figure_path, dpi=160)
    plt.close()

    print(f"Saved volatility smile to: {smile_figure_path}")
    print(f"Saved term structure to: {term_figure_path}")
    print()

    chain_30d = chain.loc[
        chain["days_to_expiry"] == 30
    ].copy()

    chain_30d["atm_distance"] = (
        chain_30d["moneyness"] - 1.0
    ).abs()

    atm_strike = chain_30d.loc[
        chain_30d["atm_distance"].idxmin(),
        "strike",
    ]

    atm_straddle = chain_30d.loc[
        chain_30d["strike"] == atm_strike
    ]

    print("ATM 30-day straddle quotes")
    print("-" * 64)

    display_columns = [
        "option_type",
        "strike",
        "implied_volatility",
        "bid",
        "ask",
        "mid",
        "delta",
        "gamma",
        "vega",
        "theta",
    ]

    print(
        atm_straddle[display_columns]
        .round(6)
        .to_string(index=False)
    )




if __name__ == "__main__":
    main()
