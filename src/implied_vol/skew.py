"""Per-expiry implied-volatility skew analysis.

Consumes the chain-level implied vols from ``src.implied_vol.solver``:

- every strike is represented once by its OTM side (``iv_source_type``);
- the curve carries strike / moneyness / log-moneyness / delta axes;
- per expiry we report ATM vol, 25-delta risk reversal and butterfly and a
  moneyness-based skew slope.

Delta convention: ``compute_delta`` returns the discounted spot delta
``D * N(d1)`` (call) / ``D * (N(d1) - 1)`` (put), the natural Black-76
quantity.  Some venues quote undiscounted forward delta (``N(d1)``); when
comparing against broker 25-delta quotes the convention must be checked
because D is close to but not exactly 1.

Interpolation never extrapolates: queries outside the observed range or
with too few points return NaN and the affected metric is explicitly
flagged instead of silently fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm

from .forward import lookup_forward_params


MIN_POINTS = 5
MIN_SIDE_POINTS = 3
MAX_VOL = 5.0


def compute_delta(
    forward: float,
    strike: float,
    time_to_expiry: float,
    sigma: float,
    discount: float,
    is_call: bool,
) -> float:
    """Black-76 discounted spot delta; NaN for unusable inputs."""
    if (
        not np.isfinite(sigma)
        or sigma <= 0
        or not np.isfinite(time_to_expiry)
        or time_to_expiry <= 0
        or forward <= 0
        or strike <= 0
    ):
        return float("nan")
    d1 = (
        np.log(forward / strike) + 0.5 * sigma**2 * time_to_expiry
    ) / (sigma * np.sqrt(time_to_expiry))
    if is_call:
        return float(discount * norm.cdf(d1))
    return float(discount * (norm.cdf(d1) - 1))


def _safe_interp(x, y, x_query):
    """Linear interpolation after NaN filtering, sorting and deduplication."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return float("nan")
    order = np.argsort(x)
    x, y = x[order], y[order]
    _, unique_index = np.unique(x, return_index=True)
    x, y = x[unique_index], y[unique_index]
    if x.size < 2:
        return float("nan")
    if x_query < x[0] or x_query > x[-1]:
        return float("nan")
    interpolator = interp1d(x, y, kind="linear")
    return float(interpolator(x_query))


def _otm_row_mask(
    frame: pd.DataFrame,
    forward: float,
) -> pd.Series:
    """Select the OTM side per strike, preferring the solver's source flag."""
    if "iv_source_type" in frame.columns:
        return frame["option_type"] == frame["iv_source_type"]
    strikes = frame["strike"].astype(float)
    expected = np.where(strikes >= forward, "call", "put")
    return pd.Series(
        frame["option_type"].to_numpy() == expected,
        index=frame.index,
    )


def build_skew_curve(
    group: pd.DataFrame,
    forward: float,
    time_to_expiry: float,
    discount: float,
) -> pd.DataFrame:
    """One row per strike on the OTM side, with delta and moneyness axes."""
    required = {"strike", "option_type", "iv_mid", "time_to_expiry"}
    missing = required - set(group.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    curve = group.copy()
    curve["iv_mid"] = pd.to_numeric(curve["iv_mid"], errors="coerce")
    curve = curve[
        curve["iv_mid"].notna()
        & np.isfinite(curve["iv_mid"].astype(float))
        & (curve["iv_mid"] > 0)
        & (curve["iv_mid"] <= MAX_VOL)
    ]
    if curve.empty:
        return curve

    curve["_otm"] = _otm_row_mask(curve, forward)
    curve = curve[curve["_otm"]].drop(columns=["_otm"])
    curve = (
        curve.sort_values("strike")
        .drop_duplicates(subset=["strike"], keep="first")
        .copy()
    )
    curve["log_moneyness"] = np.log(
        curve["strike"].astype(float) / float(forward)
    )
    curve["moneyness"] = curve["strike"].astype(float) / float(forward)
    curve["delta"] = [
        compute_delta(
            float(forward),
            float(strike),
            float(time_to_expiry),
            float(iv),
            float(discount),
            strike >= float(forward),
        )
        for strike, iv in zip(
            curve["strike"].astype(float),
            curve["iv_mid"].astype(float),
        )
    ]
    return curve.reset_index(drop=True)


def compute_skew_metrics(
    df: pd.DataFrame,
    forwards: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Per-expiry skew metrics plus full OTM curves."""
    results = []
    curves = {}
    for expiry, group in df.groupby("expiry"):
        params = lookup_forward_params(forwards, expiry)
        if params is None:
            continue
        forward, discount = params
        time_values = group["time_to_expiry"].astype(float).unique()
        if time_values.size == 0:
            continue
        time_to_expiry = float(time_values[0])

        curve = build_skew_curve(
            group,
            forward,
            time_to_expiry,
            discount,
        )
        if not curve.empty:
            curves[pd.Timestamp(expiry)] = curve

        base = {
            "expiry": pd.Timestamp(expiry),
            "forward": forward,
            "time_to_expiry": time_to_expiry,
            "num_points": int(len(curve)),
            "valid": False,
            "rr_bf_valid": False,
        }
        if len(curve) < MIN_POINTS:
            base.update(
                {
                    "atm_vol": float("nan"),
                    "vol_25c": float("nan"),
                    "vol_25p": float("nan"),
                    "rr_25": float("nan"),
                    "bf_25": float("nan"),
                    "skew_slope_90110": float("nan"),
                }
            )
            results.append(base)
            continue

        atm_vol = _safe_interp(
            curve["log_moneyness"].to_numpy(),
            curve["iv_mid"].to_numpy(),
            0.0,
        )
        calls = curve[curve["strike"].astype(float) >= forward]
        puts = curve[curve["strike"].astype(float) < forward]

        vol_25c = float("nan")
        vol_25p = float("nan")
        if len(calls) >= MIN_SIDE_POINTS:
            vol_25c = _safe_interp(
                calls["delta"].to_numpy(),
                calls["iv_mid"].to_numpy(),
                0.25,
            )
        if len(puts) >= MIN_SIDE_POINTS:
            vol_25p = _safe_interp(
                puts["delta"].to_numpy(),
                puts["iv_mid"].to_numpy(),
                -0.25,
            )

        rr_bf_ok = (
            np.isfinite(vol_25c)
            and np.isfinite(vol_25p)
            and np.isfinite(atm_vol)
        )
        rr = vol_25c - vol_25p if rr_bf_ok else float("nan")
        bf = (
            0.5 * (vol_25c + vol_25p) - atm_vol
            if rr_bf_ok
            else float("nan")
        )

        vol_90 = _safe_interp(
            curve["moneyness"].to_numpy(),
            curve["iv_mid"].to_numpy(),
            0.90,
        )
        vol_110 = _safe_interp(
            curve["moneyness"].to_numpy(),
            curve["iv_mid"].to_numpy(),
            1.10,
        )
        skew_slope = (
            vol_110 - vol_90
            if np.isfinite(vol_90) and np.isfinite(vol_110)
            else float("nan")
        )

        base.update(
            {
                "atm_vol": atm_vol,
                "vol_25c": vol_25c,
                "vol_25p": vol_25p,
                "rr_25": rr,
                "bf_25": bf,
                "skew_slope_90110": skew_slope,
                "num_calls": int(len(calls)),
                "num_puts": int(len(puts)),
                "valid": np.isfinite(atm_vol),
                "rr_bf_valid": rr_bf_ok,
            }
        )
        results.append(base)

    metrics = pd.DataFrame(results)
    if not metrics.empty:
        metrics = metrics.sort_values("time_to_expiry").reset_index(
            drop=True
        )
    return metrics, curves
