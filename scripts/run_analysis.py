"""One-command end-to-end analysis on a graded quote snapshot.

Usage:
    python scripts/run_analysis.py \
        --input outputs/real_option_chain/spy_quality_graded.csv \
        --output outputs/analysis_run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.report.pipeline import run_full_analysis  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/analysis_run")
    parser.add_argument("--ticker", default="SPY")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    quotes = pd.read_csv(
        input_path,
        parse_dates=["expiry", "snapshot_date"],
    )
    result = run_full_analysis(
        quotes,
        Path(args.output) / args.ticker,
        ticker=args.ticker,
    )

    print(f"\n=== {args.ticker} analysis complete ===")
    print(f"spot: {result.spot}")
    if result.term_structure is not None:
        print(f"term structure: {result.term_structure.shape}")
    if result.liquidity is not None:
        print(f"liquidity: {result.liquidity.overall_state}")
    if result.svi_results:
        valid = sum(item.valid for item in result.svi_results)
        print(f"SVI valid: {valid}/{len(result.svi_results)}")
    print(f"warnings: {len(result.warnings)}")
    for warning in result.warnings:
        print(f"  ! {warning}")


if __name__ == "__main__":
    main()
