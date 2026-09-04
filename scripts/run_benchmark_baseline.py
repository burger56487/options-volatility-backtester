"""Record the C++/Python Black-Scholes speedup baseline.

Mirrors the methodology in ``docs/CXX_BENCH.md``: the same fixed ATM price is
evaluated in a single-threaded Python scalar loop and through the C++ batch
kernel (when the backend is available). Results are written to
``outputs/testing/benchmark_baseline.json``. The script exits 0 even when the
backend is missing, because speedups are evidence, not a hard CI gate.
The ratio here is batch-kernel vs scalar Python (measured ~3.8x on
2026-09-04); ``docs/CXX_BENCH.md`` reports the standalone single-price
benchmark ratio (~40x) which uses a different code path.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.pricing.cpp_backend import cpp_batch_bs, is_available


def _bs_call(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    sigma_sqrt_t = volatility * math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2)
        * time_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return (
        spot
        * math.exp(-dividend_yield * time_to_expiry)
        * (0.5 * math.erfc(-d1 / math.sqrt(2.0)))
        - strike
        * math.exp(-risk_free_rate * time_to_expiry)
        * (0.5 * math.erfc(-d2 / math.sqrt(2.0)))
    )


def main() -> None:
    n = 1_000_000
    spot = strike = 100.0
    time_to_expiry, risk_free_rate = 0.5, 0.04
    dividend_yield, volatility = 0.01, 0.25

    start = time.perf_counter()
    total = 0.0
    for _ in range(n):
        total += _bs_call(
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            dividend_yield,
            volatility,
        )
    python_seconds = time.perf_counter() - start
    python_average = total / n

    baseline: dict = {
        "run_date": date.today().isoformat(),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "paths": n,
        "backend_available": is_available(),
        "python_seconds": python_seconds,
        "python_average_price": python_average,
    }
    if is_available():
        arrays = {
            key: np.full(n, value)
            for key, value in (
                ("spot", spot),
                ("strike", strike),
                ("time_to_expiry", time_to_expiry),
                ("volatility", volatility),
            )
        }
        start = time.perf_counter()
        output = cpp_batch_bs(
            arrays["spot"],
            arrays["strike"],
            arrays["time_to_expiry"],
            risk_free_rate,
            dividend_yield,
            arrays["volatility"],
            np.array(["call"] * n),
        )
        cpp_seconds = time.perf_counter() - start
        baseline["cpp_seconds"] = cpp_seconds
        baseline["cpp_average_price"] = float(output["price"][0])
        baseline["speedup"] = python_seconds / cpp_seconds
        baseline["price_absolute_difference"] = abs(
            python_average - float(output["price"][0])
        )
        baseline["methodology"] = (
            "python scalar loop vs C++ batch kernel; "
            "see docs/CXX_BENCH.md for the standalone benchmark"
        )

    output_dir = PROJECT_ROOT / "outputs" / "testing"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_baseline.json"
    output_path.write_text(
        json.dumps(baseline, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(baseline, indent=2))
    print(f"saved to {output_path}")


if __name__ == "__main__":
    main()
