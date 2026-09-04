"""Solve Black-76 implied vols on the graded real chain.

Usage:
    python scripts/run_real_chain_iv.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.implied_vol.solver import solve_chain_iv  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "real_option_chain"
GRADED_CSV = OUTPUT_DIR / "spy_quality_graded.csv"
FORWARD_CSV = OUTPUT_DIR / "spy_forward_estimates.csv"


def main() -> None:
    for path in (GRADED_CSV, FORWARD_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} missing; run the earlier scripts first."
            )
    quotes = pd.read_csv(
        GRADED_CSV,
        parse_dates=["expiry", "snapshot_date"],
    )
    forwards = pd.read_csv(FORWARD_CSV, parse_dates=["expiry"])
    solved = solve_chain_iv(quotes, forwards)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    solved.to_csv(OUTPUT_DIR / "spy_chain_iv.csv", index=False)

    status_counts = solved["iv_status"].value_counts(dropna=False).to_dict()
    payload = {
        "rows": int(len(solved)),
        "status_counts": {
            str(key): int(value) for key, value in status_counts.items()
        },
        "mid_iv_solved": int(solved["iv_mid"].notna().sum()),
        "bid_iv_solved": int(solved["iv_bid"].notna().sum()),
        "ask_iv_solved": int(solved["iv_ask"].notna().sum()),
    }
    (OUTPUT_DIR / "spy_chain_iv_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved to {OUTPUT_DIR / 'spy_chain_iv.csv'}")


if __name__ == "__main__":
    main()
