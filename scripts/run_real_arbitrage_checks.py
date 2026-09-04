"""Run the five no-arbitrage checks on the graded real chain.

Consumes ``outputs/real_option_chain/spy_quality_graded.csv`` (quality-good
quotes) and the per-expiry forward estimates from
``scripts/estimate_real_forwards.py``.

Usage:
    python scripts/run_real_arbitrage_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.arbitrage.checks import run_all_checks  # noqa: E402


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
    report = run_all_checks(quotes, forwards)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_checked": report.total_checked,
        "bound_violations": report.bound_violations,
        "bound_hard": int(
            (report.details["bounds"]["bound_severity"] == "hard").sum()
        ),
        "bound_mid_only": int(
            (report.details["bounds"]["bound_severity"] == "mid_only").sum()
        ),
        "parity_violations": report.parity_violations,
        "parity_hard": int(
            (
                report.details["parity"]["parity_severity"] == "hard"
            ).sum()
        ),
        "parity_mid_only": int(
            (
                report.details["parity"]["parity_severity"] == "mid_only"
            ).sum()
        ),
        "monotonicity_violations": report.monotonicity_violations,
        "butterfly_violations": report.butterfly_violations,
        "calendar_violations": report.calendar_violations,
        "notes": report.notes,
    }
    (OUTPUT_DIR / "spy_arbitrage_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for name, frame in report.details.items():
        frame.to_csv(
            OUTPUT_DIR / f"spy_arbitrage_{name}.csv",
            index=False,
        )

    print(
        f"checked={payload['total_checked']} "
        f"bounds(hard={payload['bound_hard']},mid={payload['bound_mid_only']}) "
        f"parity(hard={payload['parity_hard']},mid={payload['parity_mid_only']}) "
        f"mono={payload['monotonicity_violations']} "
        f"butterfly={payload['butterfly_violations']} "
        f"calendar={payload['calendar_violations']}"
    )


if __name__ == "__main__":
    main()
