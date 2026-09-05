"""Chain Greeks, heatmaps and risk concentration on the real OTM curve.

Usage:
    python scripts/run_real_chain_greeks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.greeks.chain_greeks import (  # noqa: E402
    aggregate_portfolio_greeks,
    build_greek_heatmap,
    compute_chain_greeks,
    top_risk_contracts,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "real_option_chain"
SKEW_CURVES_CSV = OUTPUT_DIR / "spy_skew_curves.csv"


def main() -> None:
    if not SKEW_CURVES_CSV.exists():
        raise FileNotFoundError(
            f"{SKEW_CURVES_CSV.name} missing; run the skew script first."
        )
    curves = pd.read_csv(SKEW_CURVES_CSV, parse_dates=["expiry"])
    greeks = compute_chain_greeks(
        curves,
        risk_free_rate=0.04,
        dividend_yield=0.012,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    greeks.to_csv(OUTPUT_DIR / "spy_chain_greeks.csv", index=False)

    for greek, option_type in [
        ("gamma", None),
        ("vega", None),
        ("delta", "call"),
        ("delta", "put"),
    ]:
        heatmap = build_greek_heatmap(
            greeks,
            greek,
            option_type=option_type,
        )
        suffix = greek if option_type is None else f"{greek}_{option_type}"
        heatmap.to_csv(
            OUTPUT_DIR / f"spy_heatmap_{suffix}.csv"
        )

    top_gamma = top_risk_contracts(greeks, "gamma", top_n=8)
    top_vega = top_risk_contracts(greeks, "vega", top_n=8)
    top_gamma.to_csv(OUTPUT_DIR / "spy_top_gamma.csv", index=False)
    top_vega.to_csv(OUTPUT_DIR / "spy_top_vega.csv", index=False)

    aggregated = aggregate_portfolio_greeks(greeks)
    payload = {
        "rows": int(len(greeks)),
        "aggregated_per_contract_mult100": aggregated,
        "top_gamma": top_gamma.to_dict(orient="records"),
        "top_vega": top_vega.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "spy_chain_greeks_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print("top gamma:")
    print(top_gamma.to_string(index=False))
    print("top vega:")
    print(top_vega.to_string(index=False))
    print("aggregated (1 contract per OTM quote, x100):")
    print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
