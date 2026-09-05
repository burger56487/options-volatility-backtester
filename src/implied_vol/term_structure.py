"""ATM implied-volatility term-structure analysis.

Consumes per-expiry skew metrics (ATM vol) and reports:

- total variance ``w = sigma_ATM^2 * T`` and its monotonicity (calendar
  no-arbitrage): total variance must not fall as maturity grows;
- forward volatility between consecutive expiries;
- shape classification with short/long-end segments;
- slope measures (absolute, annualized and relative).

Key economic point kept explicit: a *backwardated* ATM vol term structure
(near-term vol above far-term) is not itself calendar arbitrage, because
total variance can still rise with maturity.  Only a fall in total variance
is a calendar-violation signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


MIN_TTM_GAP = 1e-4
MONO_TOL = 1e-6
CLASSIFY_TOL = 0.005


@dataclass
class TermStructureResult:
    curve: pd.DataFrame
    shape: str = "insufficient_data"
    front_shape: str = ""
    back_shape: str = ""
    slope: float = float("nan")
    annualized_slope: float = float("nan")
    relative_slope: float = float("nan")
    calendar_violations: int = 0
    noise_negative_count: int = 0
    num_valid_expiries: int = 0
    warnings: list[str] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)


def _classify_shape(
    vols: np.ndarray,
    tol: float = CLASSIFY_TOL,
) -> str:
    """Classify a vol path: contango/backwardation/flat/humped/mixed."""
    if len(vols) < 2:
        return "flat"
    diffs = np.diff(vols)
    total_change = float(vols[-1] - vols[0])
    all_up = bool(np.all(diffs >= -tol))
    all_down = bool(np.all(diffs <= tol))
    if all_up and total_change > tol:
        return "contango"
    if all_down and total_change < -tol:
        return "backwardation"
    if abs(total_change) <= tol:
        return "flat"
    peak_index = int(np.argmax(vols))
    trough_index = int(np.argmin(vols))
    if 0 < peak_index < len(vols) - 1:
        return "humped"
    if 0 < trough_index < len(vols) - 1:
        return "inverted_hump"
    return "mixed"


def compute_term_structure(
    skew_metrics: pd.DataFrame,
    *,
    min_gap: float = MIN_TTM_GAP,
    mono_tol: float = MONO_TOL,
    classify_tol: float = CLASSIFY_TOL,
) -> TermStructureResult:
    """Build the ATM term structure with forward vols and shape flags."""
    required = {"expiry", "time_to_expiry", "atm_vol"}
    missing = required - set(skew_metrics.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ts = skew_metrics[
        ["expiry", "time_to_expiry", "atm_vol"]
    ].copy()
    ts["time_to_expiry"] = pd.to_numeric(
        ts["time_to_expiry"],
        errors="coerce",
    )
    ts["atm_vol"] = pd.to_numeric(ts["atm_vol"], errors="coerce")
    ts = ts[ts["atm_vol"].notna() & (ts["atm_vol"] > 0)].copy()
    ts = ts.sort_values("time_to_expiry").reset_index(drop=True)

    warnings: list[str] = []
    before_dedupe = len(ts)
    ts = ts.drop_duplicates(
        subset=["time_to_expiry"],
        keep="first",
    ).reset_index(drop=True)
    if len(ts) < before_dedupe:
        warnings.append(
            f"去重了 {before_dedupe - len(ts)} 个相同 time_to_expiry 的到期"
        )

    if len(ts) < 2:
        result = TermStructureResult(
            curve=ts,
            shape="insufficient_data",
            num_valid_expiries=int(len(ts)),
            warnings=warnings
            + ["有效到期不足 2 个，无法构造期限结构"],
        )
        return result

    ts["total_variance"] = (
        ts["atm_vol"] ** 2 * ts["time_to_expiry"]
    )
    times = ts["time_to_expiry"].to_numpy(dtype=float)
    total_var = ts["total_variance"].to_numpy(dtype=float)
    atm = ts["atm_vol"].to_numpy(dtype=float)

    n = len(ts)
    forward_vol = np.full(n, np.nan)
    fwd_var = np.full(n, np.nan)
    fwd_status = [""] * n
    calendar_violations = 0
    noise_negative_count = 0
    violations: list[dict] = []

    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt < min_gap:
            fwd_status[i] = "gap_too_small"
            warnings.append(
                f"到期间隔过小({dt:.2e})，跳过第 {i} 段前向波动率"
            )
            continue
        dw = total_var[i] - total_var[i - 1]
        segment_fwd_var = dw / dt
        fwd_var[i] = segment_fwd_var
        if segment_fwd_var < -mono_tol:
            calendar_violations += 1
            fwd_status[i] = "calendar_violation"
            violations.append(
                {
                    "segment": i,
                    "t1": float(times[i - 1]),
                    "t2": float(times[i]),
                    "w1": float(total_var[i - 1]),
                    "w2": float(total_var[i]),
                    "fwd_var": float(segment_fwd_var),
                }
            )
            warnings.append(
                f"总方差在 T={times[i]:.4f} 处递减 "
                f"(w: {total_var[i - 1]:.3e} -> "
                f"{total_var[i]:.3e})，存在日历套利"
            )
        elif segment_fwd_var < 0:
            noise_negative_count += 1
            fwd_status[i] = "noise_negative"
            forward_vol[i] = 0.0
        else:
            fwd_status[i] = "ok"
            forward_vol[i] = float(np.sqrt(segment_fwd_var))

    ts["fwd_var"] = fwd_var
    ts["forward_vol"] = forward_vol
    ts["fwd_status"] = fwd_status

    slope = float(atm[-1] - atm[0])
    span = float(times[-1] - times[0])
    annualized_slope = slope / span if span > 0 else float("nan")
    relative_slope = slope / float(atm[0]) if atm[0] > 0 else float("nan")

    shape = _classify_shape(atm, tol=classify_tol)
    mid = len(atm) // 2
    front_shape = (
        _classify_shape(atm[: mid + 1], tol=classify_tol)
        if mid >= 1
        else ""
    )
    back_shape = (
        _classify_shape(atm[mid:], tol=classify_tol)
        if len(atm) - mid >= 2
        else ""
    )

    return TermStructureResult(
        curve=ts,
        shape=shape,
        front_shape=front_shape,
        back_shape=back_shape,
        slope=slope,
        annualized_slope=annualized_slope,
        relative_slope=relative_slope,
        calendar_violations=calendar_violations,
        noise_negative_count=noise_negative_count,
        num_valid_expiries=n,
        warnings=warnings,
        violations=violations,
    )
