"""Verify C++ kernels against Python and benchmark batch speedups."""

from __future__ import annotations

import ctypes
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.pricing.black_scholes import price_and_greeks
from src.pricing.implied_volatility import implied_volatility


def load_lib() -> ctypes.CDLL:
    candidates = [
        Path("outputs/bs_kernels.dll"),
        PROJECT_ROOT / "outputs/bs_kernels.dll",
    ]
    lib = ctypes.CDLL(str(next(path for path in candidates if path.exists())))
    lib.batch_bs.restype = None
    lib.batch_iv.restype = ctypes.c_int
    return lib


def to_ptr(array: np.ndarray) -> ctypes.c_void_p:
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def main() -> None:
    lib = load_lib()
    rng = np.random.default_rng(7)
    n = 200_000
    spot = rng.uniform(50, 200, n)
    strike = rng.uniform(50, 200, n)
    t = rng.uniform(0.05, 2.0, n)
    r = np.full(n, 0.04)
    q = np.full(n, 0.01)
    vol = rng.uniform(0.1, 0.8, n)
    call = (rng.random(n) < 0.5).astype(np.int32)

    out = {
        "price": np.zeros(n),
        "delta": np.zeros(n),
        "gamma": np.zeros(n),
        "vega": np.zeros(n),
        "theta": np.zeros(n),
        "rho": np.zeros(n),
    }

    def run_cpp():
        lib.batch_bs(
            to_ptr(spot),
            to_ptr(strike),
            to_ptr(t),
            to_ptr(r),
            to_ptr(q),
            to_ptr(vol),
            call.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n,
            to_ptr(out["price"]),
            to_ptr(out["delta"]),
            to_ptr(out["gamma"]),
            to_ptr(out["vega"]),
            to_ptr(out["theta"]),
            to_ptr(out["rho"]),
        )

    start = time.perf_counter()
    run_cpp()
    cpp_seconds = time.perf_counter() - start

    start = time.perf_counter()
    py = {
        "price": np.zeros(n),
        "delta": np.zeros(n),
        "gamma": np.zeros(n),
        "vega": np.zeros(n),
        "theta": np.zeros(n),
        "rho": np.zeros(n),
    }
    for i in range(n):
        result = price_and_greeks(
            spot=float(spot[i]),
            strike=float(strike[i]),
            time_to_expiry=float(t[i]),
            risk_free_rate=float(r[i]),
            volatility=float(vol[i]),
            option_type="call" if call[i] else "put",
            dividend_yield=float(q[i]),
        )
        py["price"][i] = result.price
        py["delta"][i] = result.delta
        py["gamma"][i] = result.gamma
        py["vega"][i] = result.vega
        py["theta"][i] = result.theta
        py["rho"][i] = result.rho
    python_seconds = time.perf_counter() - start

    diffs = {
        key: float(np.max(np.abs(out[key] - py[key])))
        for key in py
    }

    # IV batch on a smaller sample (Python IV loop is slow).
    n_iv = 3_000
    spot_iv = rng.uniform(50, 200, n_iv)
    moneyness_iv = rng.uniform(-0.15, 0.15, n_iv)
    strike_iv = spot_iv * np.exp(moneyness_iv)
    t_iv = rng.uniform(0.05, 2.0, n_iv)
    vol_iv = rng.uniform(0.1, 0.8, n_iv)
    call_iv = (rng.random(n_iv) < 0.5).astype(np.int32)
    mids = np.zeros(n_iv)
    for i in range(n_iv):
        mids[i] = price_and_greeks(
            spot=float(spot_iv[i]),
            strike=float(strike_iv[i]),
            time_to_expiry=float(t_iv[i]),
            risk_free_rate=0.04,
            volatility=float(vol_iv[i]),
            option_type="call" if call_iv[i] else "put",
            dividend_yield=0.01,
        ).price
    iv_out = np.zeros(n_iv)

    def run_iv_cpp():
        lib.batch_iv(
            to_ptr(mids),
            to_ptr(spot_iv),
            to_ptr(strike_iv),
            to_ptr(t_iv),
            to_ptr(np.full(n_iv, 0.04)),
            to_ptr(np.full(n_iv, 0.01)),
            call_iv.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_iv,
            ctypes.c_double(1e-8),
            ctypes.c_int(100),
            to_ptr(iv_out),
        )

    start = time.perf_counter()
    run_iv_cpp()
    iv_cpp_seconds = time.perf_counter() - start

    start = time.perf_counter()
    iv_py = np.zeros(n_iv)
    for i in range(n_iv):
        iv_py[i] = implied_volatility(
            market_price=float(mids[i]),
            spot=float(spot_iv[i]),
            strike=float(strike_iv[i]),
            time_to_expiry=float(t_iv[i]),
            risk_free_rate=0.04,
            option_type="call" if call_iv[i] else "put",
            dividend_yield=0.01,
        )
    iv_python_seconds = time.perf_counter() - start
    finite = np.isfinite(iv_out) & np.isfinite(iv_py)
    iv_diff = (
        float(np.max(np.abs(iv_out[finite] - iv_py[finite])))
        if finite.any()
        else float("nan")
    )
    iv_nan_cpp = int((~np.isfinite(iv_out)).sum())
    iv_nan_python = int((~np.isfinite(iv_py)).sum())

    evidence = {
        "date": "2026-09-04",
        "n_batch": n,
        "cpp_bs_seconds": cpp_seconds,
        "python_bs_seconds": python_seconds,
        "bs_speedup": python_seconds / cpp_seconds,
        "max_abs_diff": diffs,
        "n_iv": n_iv,
        "cpp_iv_seconds": iv_cpp_seconds,
        "python_iv_seconds": iv_python_seconds,
        "iv_speedup": iv_python_seconds / iv_cpp_seconds,
        "iv_max_abs_diff": iv_diff,
        "iv_nan_cpp": iv_nan_cpp,
        "iv_nan_python": iv_nan_python,
    }
    output = Path("outputs") / "cpp_evidence.json"
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
