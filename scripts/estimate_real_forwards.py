"""Estimate per-expiry implied forwards/rates on the graded real chain.

Reads ``outputs/real_option_chain/spy_quality_graded.csv`` (all quotes with
five-class quality labels), keeps quality-good pairs and runs the robust
weighted put-call parity regression for every expiry.

Usage:
    python scripts/estimate_real_forwards.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.implied_vol.forward import estimate_all_forwards  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "real_option_chain"
GRADED_CSV = OUTPUT_DIR / "spy_quality_graded.csv"
ESTIMATES_CSV = OUTPUT_DIR / "spy_forward_estimates.csv"
ESTIMATES_JSON = OUTPUT_DIR / "spy_forward_estimates.json"


def main() -> None:
    if not GRADED_CSV.exists():
        raise FileNotFoundError(
            f"Graded chain not found: {GRADED_CSV}. "
            "Run scripts/grade_real_option_chain.py first."
        )
    frame = pd.read_csv(GRADED_CSV, parse_dates=["expiry", "snapshot_date"])
    estimates = estimate_all_forwards(frame)
    if estimates.empty:
        raise RuntimeError("No call/put pairs survived the quality screen.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(ESTIMATES_CSV, index=False)

    valid_rows = estimates[estimates["valid"]]
    summary = {
        "total_expiries": int(len(estimates)),
        "valid_expiries": int(len(valid_rows)),
        "expiries": estimates["expiry"].tolist(),
        "pass_rate": (
            round(100.0 * len(valid_rows) / len(estimates), 1)
            if len(estimates)
            else 0.0
        ),
    }
    ESTIMATES_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"total expiries: {summary['total_expiries']}, "
        f"valid: {summary['valid_expiries']} "
        f"({summary['pass_rate']}%)"
    )
    print(estimates.to_string(index=False))
    print(f"saved to {ESTIMATES_CSV}")


if __name__ == "__main__":
    main()
