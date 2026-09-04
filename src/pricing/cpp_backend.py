"""Optional C++ backend switch for batch pricing and Monte Carlo.

Loads ``outputs/bs_kernels.dll`` on Windows or ``outputs/bs_kernels.so`` on
Linux (built from ``cpp/src/bs_kernels.cpp`` with zig or any C++ compiler).
If unavailable, functions fall back to Python so the rest of the project
keeps working.
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np


def _locate_library() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    for name in ("bs_kernels.dll", "bs_kernels.so"):
        for base in (Path("outputs"), root / "outputs"):
            path = base / name
            if path.exists():
                return path
    return None


_LIB = None
if _locate_library() is not None:
    try:
        _LIB = ctypes.CDLL(str(_locate_library()))
        _LIB.batch_bs.restype = None
        _LIB.mc_gbm.restype = None
        _LIB.batch_iv.restype = ctypes.c_int
        _LIB.scenario_pnl.restype = None
        _LIB.portfolio_var.restype = None
        _LIB.mc_gbm_parallel.restype = None
    except OSError:
        _LIB = None


def is_available() -> bool:
    return _LIB is not None


def _ptr(array: np.ndarray):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def cpp_batch_bs(
    spot,
    strike,
    time_to_expiry,
    risk_free_rate,
    dividend_yield,
    volatility,
    option_type,
):
    """C++ batch BSM price/Greeks; raises RuntimeError when DLL is absent."""
    if _LIB is None:
        raise RuntimeError("C++ backend not available; rebuild bs_kernels.dll.")
    arrays = [
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(time_to_expiry, dtype=float),
    ]
    n = len(arrays[0])
    arrays += [
        np.full(n, float(risk_free_rate)),
        np.full(n, float(dividend_yield)),
        np.asarray(volatility, dtype=float),
    ]
    if len(arrays[5]) == 1:
        arrays[5] = np.full(n, float(volatility))
    call = (np.asarray(option_type) == "call").astype(np.int32)
    outputs = {
        key: np.zeros(n)
        for key in ("price", "delta", "gamma", "vega", "theta", "rho")
    }
    _LIB.batch_bs(
        _ptr(arrays[0]),
        _ptr(arrays[1]),
        _ptr(arrays[2]),
        _ptr(arrays[3]),
        _ptr(arrays[4]),
        _ptr(arrays[5]),
        call.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n,
        _ptr(outputs["price"]),
        _ptr(outputs["delta"]),
        _ptr(outputs["gamma"]),
        _ptr(outputs["vega"]),
        _ptr(outputs["theta"]),
        _ptr(outputs["rho"]),
    )
    return outputs


def cpp_mc_gbm(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    n_paths: int,
    option_type: str,
    seed: int,
) -> dict:
    """C++ exact-GBM Monte Carlo price and standard error."""
    if _LIB is None:
        raise RuntimeError("C++ backend not available; rebuild bs_kernels.dll.")
    price = ctypes.c_double()
    se = ctypes.c_double()
    _LIB.mc_gbm(
        ctypes.c_double(float(spot)),
        ctypes.c_double(float(strike)),
        ctypes.c_double(float(time_to_expiry)),
        ctypes.c_double(float(risk_free_rate)),
        ctypes.c_double(float(dividend_yield)),
        ctypes.c_double(float(volatility)),
        ctypes.c_int(int(n_paths)),
        ctypes.c_uint(int(seed)),
        ctypes.c_int(1 if option_type == "call" else 0),
        ctypes.byref(price),
        ctypes.byref(se),
    )
    return {"price": float(price.value), "standard_error": float(se.value)}


def cpp_mc_gbm_parallel(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    n_paths: int,
    option_type: str,
    seed: int,
    threads: int,
) -> dict:
    if _LIB is None:
        raise RuntimeError("C++ backend not available.")
    price = ctypes.c_double()
    se = ctypes.c_double()
    _LIB.mc_gbm_parallel(
        ctypes.c_double(float(spot)),
        ctypes.c_double(float(strike)),
        ctypes.c_double(float(time_to_expiry)),
        ctypes.c_double(float(risk_free_rate)),
        ctypes.c_double(float(dividend_yield)),
        ctypes.c_double(float(volatility)),
        ctypes.c_int(int(n_paths)),
        ctypes.c_uint(int(seed)),
        ctypes.c_int(1 if option_type == "call" else 0),
        ctypes.c_int(int(threads)),
        ctypes.byref(price),
        ctypes.byref(se),
    )
    return {"price": float(price.value), "standard_error": float(se.value)}


def cpp_scenario_pnl(
    spot,
    strike,
    time_to_expiry,
    risk_free_rate,
    dividend_yield,
    volatility,
    option_type,
    spot_shock: float,
    vol_shock: float,
) -> np.ndarray:
    """Per-option PnL under a spot/vol shock scenario."""
    if _LIB is None:
        raise RuntimeError("C++ backend not available.")
    n = len(spot)
    call = (np.asarray(option_type) == "call").astype(np.int32)
    out = np.zeros(n)
    _LIB.scenario_pnl(
        _ptr(np.asarray(spot, dtype=float)),
        _ptr(np.asarray(strike, dtype=float)),
        _ptr(np.asarray(time_to_expiry, dtype=float)),
        _ptr(np.full(n, float(risk_free_rate))),
        _ptr(np.full(n, float(dividend_yield))),
        _ptr(np.asarray(volatility, dtype=float)),
        call.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n,
        ctypes.c_double(float(spot_shock)),
        ctypes.c_double(float(vol_shock)),
        _ptr(out),
    )
    return out


def cpp_portfolio_var(
    exposures: np.ndarray,
    covariance: np.ndarray,
    z_score: float = 1.645,
) -> dict:
    """Portfolio VaR and Euler contributions computed in C++."""
    if _LIB is None:
        raise RuntimeError("C++ backend not available.")
    exposures = np.asarray(exposures, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n = len(exposures)
    var = ctypes.c_double()
    contributions = np.zeros(n)
    _LIB.portfolio_var(
        _ptr(exposures),
        _ptr(covariance),
        n,
        ctypes.c_double(float(z_score)),
        ctypes.byref(var),
        _ptr(contributions),
    )
    return {"var": float(var.value), "contributions": contributions}


def cpp_surface_prices(
    surface,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    moneyness_grid=None,
) -> dict:
    """Reprice vanilla options from every surface node via the C++ kernel."""
    import math

    import numpy as np

    if moneyness_grid is None:
        moneyness_grid = np.linspace(-0.1, 0.1, 9)
    rows = []
    for point in surface.points:
        for k in moneyness_grid:
            iv = surface.interpolate_iv(float(k), point.time_to_expiry)
            rows.append(
                {
                    "spot": spot,
                    "strike": spot * math.exp(float(k)),
                    "t": point.time_to_expiry,
                    "vol": iv,
                    "call": True,
                }
            )
    frame = {
        "spot": np.array([row["spot"] for row in rows]),
        "strike": np.array([row["strike"] for row in rows]),
        "t": np.array([row["t"] for row in rows]),
        "vol": np.array([row["vol"] for row in rows]),
        "call": np.array(["call"] * len(rows)),
    }
    out = cpp_batch_bs(
        frame["spot"],
        frame["strike"],
        frame["t"],
        risk_free_rate,
        dividend_yield,
        frame["vol"],
        frame["call"],
    )
    out["surface_points"] = rows
    return out
