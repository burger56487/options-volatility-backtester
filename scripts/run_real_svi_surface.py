"""Robust SVI calibration and surface no-arbitrage screens on real data.

Usage:
    python scripts/run_real_svi_surface.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.volatility_surface.svi_analysis import (  # noqa: E402
    SVIResult,
    calibrate_svi_curve,
    check_calendar_arbitrage,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "real_option_chain"
SKEW_CURVES_CSV = OUTPUT_DIR / "spy_skew_curves.csv"


def main() -> None:
    if not SKEW_CURVES_CSV.exists():
        raise FileNotFoundError(
            f"{SKEW_CURVES_CSV.name} missing; run the skew script first."
        )
    curves = pd.read_csv(SKEW_CURVES_CSV, parse_dates=["expiry"])
    results: list[SVIResult] = []
    rows = []
    for expiry, group in curves.groupby("expiry"):
        time_to_expiry = float(
            group["time_to_expiry"].dropna().iloc[0]
        )
        result = calibrate_svi_curve(
            group,
            expiry,
            time_to_expiry,
        )
        results.append(result)
        rows.append(
            {
                "expiry": pd.Timestamp(expiry).date().isoformat(),
                "time_to_expiry": time_to_expiry,
                "num_points": result.num_points,
                "rmse_vol": result.rmse_vol,
                "rmse_total_var": result.rmse_total_var,
                "butterfly_violations": result.butterfly_violations,
                "min_g": result.min_g,
                "min_w": result.min_w,
                "converged": result.converged,
                "valid": result.valid,
                "params": (
                    result.params.tolist()
                    if result.converged
                    else None
                ),
                "warnings": "; ".join(result.warnings),
            }
        )

    calendar_violations, calendar_details = check_calendar_arbitrage(
        results
    )
    frame = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "spy_svi_surface_results.csv", index=False)
    payload = {
        "expiries": rows,
        "calendar_violations": calendar_violations,
        "calendar_details": calendar_details,
        "summary": {
            "total_expiries": int(len(results)),
            "valid_expiries": int(sum(r.valid for r in results)),
        },
    }
    (OUTPUT_DIR / "spy_svi_surface_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(
        frame[
            [
                "expiry",
                "num_points",
                "rmse_vol",
                "rmse_total_var",
                "butterfly_violations",
                "min_g",
                "valid",
            ]
        ].to_string(index=False)
    )
    print(
        f"calendar violations: {calendar_violations} "
        f"({len(calendar_details)} pairs)"
    )


if __name__ == "__main__":
    main()
