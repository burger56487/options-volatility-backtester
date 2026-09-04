"""Per-expiry skew metrics on the Black-76 chain IVs.

Usage:
    python scripts/run_real_skew.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.implied_vol.skew import compute_skew_metrics  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "real_option_chain"
CHAIN_IV_CSV = OUTPUT_DIR / "spy_chain_iv.csv"
FORWARD_CSV = OUTPUT_DIR / "spy_forward_estimates.csv"


def _plot_curves(curves: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 6))
    for expiry, curve in sorted(
        curves.items(),
        key=lambda item: float(item[1]["time_to_expiry"].iloc[0]),
    ):
        axis.scatter(
            curve["log_moneyness"],
            curve["iv_mid"],
            s=18,
            label=pd.Timestamp(expiry).date().isoformat(),
        )
    axis.axvline(0.0, color="grey", linewidth=0.8, linestyle="--")
    axis.set_xlabel("log moneyness ln(K/F)")
    axis.set_ylabel("Black-76 implied volatility (OTM side)")
    axis.set_title("SPY OTM implied-volatility skew by expiry (2026-09-04)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = OUTPUT_DIR / "spy_skew_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    for path in (CHAIN_IV_CSV, FORWARD_CSV):
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} missing; run the earlier scripts first."
            )
    chain = pd.read_csv(CHAIN_IV_CSV, parse_dates=["expiry"])
    forwards = pd.read_csv(FORWARD_CSV, parse_dates=["expiry"])
    metrics, curves = compute_skew_metrics(chain, forwards)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "spy_skew_metrics.csv", index=False)
    if curves:
        all_curves = pd.concat(
            [curve.assign(expiry=expiry) for expiry, curve in curves.items()],
            ignore_index=True,
        )
        all_curves.to_csv(OUTPUT_DIR / "spy_skew_curves.csv", index=False)
    figure_path = _plot_curves(curves)

    payload = metrics.to_dict(orient="records")
    (OUTPUT_DIR / "spy_skew_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print(f"saved metrics/curves/figure to {OUTPUT_DIR}")
    print(f"figure: {figure_path.name}")


if __name__ == "__main__":
    main()
