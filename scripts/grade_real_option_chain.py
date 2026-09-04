"""Grade the real SPY chain snapshot with five-class quality labels.

Reads the analysed real-chain output (all quotes, including quotes excluded
from the IV universe), adds a per-row ``quality`` label without dropping any
data, writes the graded frame and a machine-readable grading report.

Usage:
    python scripts/grade_real_option_chain.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.market_data.quality_grading import (  # noqa: E402
    clean_graded_subset,
    generate_grading_report,
    grade_quote_quality,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "real_option_chain"
ANALYSED_CSV = OUTPUT_DIR / "spy_option_chain_analysed.csv"
GRADED_CSV = OUTPUT_DIR / "spy_quality_graded.csv"
GRADED_REPORT = OUTPUT_DIR / "spy_grading_report.json"
CLEAN_PARQUET = OUTPUT_DIR / "cleaned_chain.parquet"


def main() -> None:
    if not ANALYSED_CSV.exists():
        raise FileNotFoundError(
            f"Analysed chain not found: {ANALYSED_CSV}. "
            "Run scripts/download_real_spy_option_snapshot.py first."
        )

    frame = pd.read_csv(ANALYSED_CSV)
    graded = grade_quote_quality(frame)
    clean = clean_graded_subset(graded)
    report = generate_grading_report(graded)
    report["source"] = ANALYSED_CSV.name
    report["clean_subset_rows"] = int(len(clean))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graded.to_csv(GRADED_CSV, index=False)

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        report["parquet_export"] = "skipped: pyarrow not installed"
    else:
        clean.to_parquet(CLEAN_PARQUET, index=False)
        report["parquet_export"] = CLEAN_PARQUET.name

    GRADED_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    by_quality = report["by_quality"]
    print(
        f"总合约 {report['total_contracts']}，good {by_quality['good']} "
        f"({report['good_pct']}%)，clean subset {report['clean_subset_rows']}"
    )
    print(f"graded csv: {GRADED_CSV}")
    print(f"report   : {GRADED_REPORT}")


if __name__ == "__main__":
    main()
