"""ATM term structure and forward vol on the real skew metrics.

Usage:
    python scripts/run_real_term_structure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.implied_vol.term_structure import (  # noqa: E402
    compute_term_structure,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "real_option_chain"
SKEW_METRICS_CSV = OUTPUT_DIR / "spy_skew_metrics.csv"


def _plot(result, output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    curve = result.curve
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        curve["time_to_expiry"],
        curve["atm_vol"],
        marker="o",
        label="ATM vol",
    )
    valid_fwd = curve.dropna(subset=["forward_vol"])
    if not valid_fwd.empty:
        axis.plot(
            valid_fwd["time_to_expiry"],
            valid_fwd["forward_vol"],
            marker="s",
            linestyle="--",
            label="forward vol",
        )
    axis.set_xlabel("time to expiry (years)")
    axis.set_ylabel("volatility")
    axis.set_title("SPY ATM term structure (2026-09-04)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "spy_term_structure.png", dpi=150)
    plt.close(fig)


def main() -> None:
    if not SKEW_METRICS_CSV.exists():
        raise FileNotFoundError(
            f"{SKEW_METRICS_CSV.name} missing; run the skew script first."
        )
    skew_metrics = pd.read_csv(SKEW_METRICS_CSV)
    result = compute_term_structure(skew_metrics)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.curve.to_csv(OUTPUT_DIR / "spy_term_structure.csv", index=False)
    _plot(result, OUTPUT_DIR)

    payload = {
        "shape": result.shape,
        "front_shape": result.front_shape,
        "back_shape": result.back_shape,
        "slope": result.slope,
        "annualized_slope": result.annualized_slope,
        "relative_slope": result.relative_slope,
        "calendar_violations": result.calendar_violations,
        "noise_negative_count": result.noise_negative_count,
        "num_valid_expiries": result.num_valid_expiries,
        "warnings": result.warnings,
        "violations": result.violations,
    }
    (OUTPUT_DIR / "spy_term_structure_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(result.curve.to_string(index=False))


if __name__ == "__main__":
    main()
