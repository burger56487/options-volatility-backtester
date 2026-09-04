"""Download a real SPY option-chain snapshot, clean it, solve IV, plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.market_data.real_option_chain import (
    add_implied_volatility,
    clean_quote_frame,
    fetch_option_chain_snapshot,
)


def _plot_surface(quotes: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    valid = quotes.dropna(subset=["iv"])
    valid = valid[valid["iv"] > 0]

    # Skew for the most populated expiry.
    counts = valid["expiry"].value_counts()
    target_expiry = counts.index[0]
    skew = valid[valid["expiry"] == target_expiry]

    fig, axis = plt.subplots(figsize=(9, 6))
    for option_type in ("call", "put"):
        subset = skew[skew["option_type"] == option_type]
        axis.scatter(
            subset["log_moneyness"],
            subset["iv"],
            label=option_type,
            alpha=0.7,
            s=14,
        )
    axis.set_xlabel("log(K / F)")
    axis.set_ylabel("Implied volatility")
    axis.set_title(
        f"SPY implied-volatility skew  {target_expiry.date()}"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "spy_iv_skew.png", dpi=160)
    plt.close(fig)

    # Term structure: ATM IV per expiry.
    records = []
    for expiry, group in valid.groupby("expiry"):
        atm = group.loc[
            group["log_moneyness"].abs().idxmin()
        ]
        records.append(
            {
                "expiry": expiry,
                "atm_iv": atm["iv"],
            }
        )
    term = pd.DataFrame(records).sort_values("expiry")

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(term["expiry"], term["atm_iv"], marker="o")
    axis.set_xlabel("Expiry")
    axis.set_ylabel("ATM implied volatility")
    axis.set_title("SPY implied-volatility term structure")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "spy_iv_term_structure.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and analyse a real SPY option-chain snapshot."
    )
    parser.add_argument(
        "--symbol",
        default="SPY",
    )
    parser.add_argument(
        "--max-expiries",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--dividend-yield",
        type=float,
        default=None,
        help="Override dividend yield; default estimates it from the chain.",
    )
    args = parser.parse_args()

    quotes, spot = fetch_option_chain_snapshot(
        symbol=args.symbol,
        max_expiries=args.max_expiries,
    )

    cleaned, report = clean_quote_frame(quotes)
    analysed = add_implied_volatility(
        cleaned,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
    )

    # Illiquid deep-ITM/OTM quotes produce noisy implied vols (stale bid/ask,
    # zero open interest). Analysis and charts use the liquid subset.
    active = analysed[
        (analysed["open_interest"].fillna(0) > 0)
        & (analysed["log_moneyness"].abs() <= 0.15)
    ].copy()

    output = Path("outputs/real_option_chain")
    output.mkdir(parents=True, exist_ok=True)

    analysed.to_csv(
        output / "spy_option_chain_analysed.csv",
        index=False,
    )
    active.to_csv(
        output / "spy_option_chain_active.csv",
        index=False,
    )

    no_arb_violations = int(
        (~active["no_arbitrage_pass"]).sum()
    )
    solved = int(active["iv"].notna().sum())
    summary = {
        "symbol": args.symbol,
        "spot": spot,
        "snapshot_date": str(
            analysed["snapshot_date"].iloc[0].date()
        ),
        "quality_report": report,
        "no_arbitrage_violations": no_arb_violations,
        "active_rows": int(len(active)),
        "iv_solved_rows_in_active": solved,
        "iv_failed_rows_in_active": int(
            (active["iv_error"] != "").sum()
        ),
        "risk_free_rate": args.risk_free_rate,
        "dividend_yield": args.dividend_yield,
        "boundary": (
            "real option-chain snapshot; European IV solved on mid "
            "quotes; not a strategy backtest; active subset = "
            "open_interest > 0 and |log moneyness| <= 0.15"
        ),
    }

    with open(
        output / "spy_quality_report.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    _plot_surface(active, output)

    print(f"spot: {spot:.2f}")
    print(f"raw rows: {report['total_rows']}, "
          f"kept: {report['kept_rows']}, "
          f"retention: {report['retention_rate']:.3f}")
    print(f"active rows: {len(active)}")
    print(f"no-arb violations: {no_arb_violations}")
    print(f"IV solved (active): {solved}/{len(active)}")
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
