"""Five-class no-arbitrage checks on an option chain.

Checks use the per-expiry implied forward ``F`` and discount factor ``D``
from ``src.implied_vol.forward``:

1. price bounds (European, expressed in F/D);
2. put-call parity deviation;
3. cross-strike monotonicity (calls decrease, puts increase in K);
4. butterfly convexity (slope form, valid for uneven strike spacing);
5. calendar monotonicity (longer expiry >= shorter, same K/type).

Two disciplines that differ from a naive implementation:

- With bid/ask quotes available, bound/parity checks report ``hard``
  violations (tradable at the quotes) and ``mid_only`` diagnostics
  separately instead of relying on a fixed relative tolerance.
- The calendar check is only a true no-arbitrage screen for American
  options or European options without dividends in the interval.  SPY pays
  dividends, so calendar violations are reported as diagnostics and the
  caller should interpret them with that caveat.

The F/D estimates are themselves fitted from the same mid quotes, so bound
and parity "violations" are partly residuals of that fit; they are most
useful for locating quote-level outliers, not for proving tradable profit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


GOOD_QUALITY = "good"
ABS_TOL = 0.02
REL_TOL = 0.01


@dataclass
class ArbitrageReport:
    """Counts plus per-contract detail frames for every check."""

    bound_violations: int = 0
    parity_violations: int = 0
    monotonicity_violations: int = 0
    butterfly_violations: int = 0
    calendar_violations: int = 0
    total_checked: int = 0
    notes: list[str] = field(default_factory=list)
    details: dict[str, pd.DataFrame] = field(default_factory=dict)


def _quality_frame(df: pd.DataFrame, good_only: bool) -> pd.DataFrame:
    work = df.copy()
    if good_only:
        if "quality" not in work.columns:
            raise ValueError(
                "good_only=True requires a 'quality' column; "
                "pass good_only=False for unlabelled chains."
            )
        work = work[work["quality"] == GOOD_QUALITY]
    return work


def _normalise_forwards(forwards: pd.DataFrame) -> pd.DataFrame:
    """Coerce expiry dtype and keep only valid rows for safe lookups."""
    frame = forwards.copy()
    if "expiry" in frame.columns:
        frame["expiry"] = pd.to_datetime(frame["expiry"])
    if "valid" in frame.columns:
        frame = frame[frame["valid"].astype(bool)]
    return frame


def _params_for_expiry(
    forwards: pd.DataFrame,
    expiry,
) -> tuple[float, float] | None:
    """Return (F, D) for an expiry or None when no valid estimate exists."""
    rows = forwards[pd.to_datetime(forwards["expiry"]) == pd.Timestamp(expiry)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    forward = float(row["forward"])
    discount = float(row["discount_factor"])
    if not np.isfinite(forward) or not np.isfinite(discount):
        return None
    return forward, discount


def _bound_pair(
    option_type: pd.Series,
    forward: float,
    discount: float,
    strike: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    is_call = (option_type == "call").to_numpy()
    intrinsic = np.where(
        is_call,
        discount * np.maximum(forward - strike, 0.0),
        discount * np.maximum(strike - forward, 0.0),
    )
    upper = np.where(is_call, discount * forward, discount * strike)
    return intrinsic, upper


def check_bounds(
    df: pd.DataFrame,
    forwards: pd.DataFrame,
    good_only: bool = True,
    abs_tol: float = ABS_TOL,
) -> pd.DataFrame:
    """European F/D price-bound check; marks hard/mid-only violations."""
    work = _quality_frame(df, good_only)
    required = {
        "expiry",
        "strike",
        "option_type",
        "mid",
    }
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    work["bound_ok"] = pd.Series(True, index=work.index, dtype=object)
    work["bound_severity"] = "ok"
    work["bound_reason"] = ""
    has_quotes = {"bid", "ask"}.issubset(work.columns)
    forwards = _normalise_forwards(forwards)

    for expiry, group in work.groupby("expiry"):
        params = _params_for_expiry(forwards, expiry)
        if params is None:
            work.loc[group.index, "bound_ok"] = np.nan
            work.loc[group.index, "bound_severity"] = "no_valid_forward"
            work.loc[group.index, "bound_reason"] = "no_valid_forward"
            continue
        forward, discount = params
        lower, upper = _bound_pair(
            group["option_type"],
            forward,
            discount,
            group["strike"].to_numpy(dtype=float),
        )
        mid = group["mid"].to_numpy(dtype=float)
        if has_quotes:
            bid = group["bid"].to_numpy(dtype=float)
            ask = group["ask"].to_numpy(dtype=float)
            hard_below = ask < lower - abs_tol
            hard_above = bid > upper + abs_tol
            hard = hard_below | hard_above
            mid_only = ~hard & (
                (mid < lower - abs_tol) | (mid > upper + abs_tol)
            )
            ok = ~hard & ~mid_only
            reason = np.where(
                hard_below,
                "hard_below_lower_bound",
                np.where(
                    hard_above,
                    "hard_above_upper_bound",
                    np.where(
                        mid_only,
                        "mid_only_out_of_bounds",
                        "",
                    ),
                ),
            )
            severity = np.where(
                hard,
                "hard",
                np.where(mid_only, "mid_only", "ok"),
            )
        else:
            scale = np.maximum.reduce(
                [np.abs(lower), np.abs(upper), np.ones_like(mid)]
            )
            ok = (
                (mid >= lower - abs_tol - REL_TOL * scale)
                & (mid <= upper + abs_tol + REL_TOL * scale)
            )
            reason = np.where(ok, "", "out_of_bounds")
            severity = np.where(ok, "ok", "mid_only")

        work.loc[group.index, "bound_ok"] = ok
        work.loc[group.index, "bound_severity"] = severity
        work.loc[group.index[~np.asarray(ok, dtype=bool)], "bound_reason"] = (
            reason[~np.asarray(ok, dtype=bool)]
        )
    return work


def _parity_ok_and_severity(
    target: np.ndarray,
    mid_diff: np.ndarray,
    call_bid: np.ndarray | None,
    call_ask: np.ndarray | None,
    put_bid: np.ndarray | None,
    put_ask: np.ndarray | None,
    abs_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        call_bid is not None
        and call_ask is not None
        and put_bid is not None
        and put_ask is not None
    ):
        low = call_bid - put_ask
        high = call_ask - put_bid
        hard = (target < low - abs_tol) | (target > high + abs_tol)
        mid_only = ~hard & (
            (target < mid_diff - abs_tol) | (target > mid_diff + abs_tol)
        )
        ok = ~hard & ~mid_only
        severity = np.where(
            hard,
            "hard",
            np.where(mid_only, "mid_only", "ok"),
        )
        return ok, severity, hard | mid_only
    scale = np.maximum(np.abs(target), 1.0)
    ok = np.abs(mid_diff - target) <= abs_tol + REL_TOL * scale
    severity = np.where(ok, "ok", "mid_only")
    return ok, severity, ~ok


def check_parity(
    df: pd.DataFrame,
    forwards: pd.DataFrame,
    good_only: bool = True,
    abs_tol: float = ABS_TOL,
) -> tuple[int, pd.DataFrame]:
    """Put-call parity deviation against the implied F/D."""
    work = _quality_frame(df, good_only)
    call_cols = ["expiry", "strike", "mid"]
    put_cols = ["expiry", "strike", "mid"]
    quote_cols = {"bid", "ask"}
    if quote_cols.issubset(work.columns):
        call_cols += ["bid", "ask"]
        put_cols += ["bid", "ask"]
    calls = work[work["option_type"] == "call"][call_cols].rename(
        columns={
            "mid": "mid_c",
            "bid": "bid_c",
            "ask": "ask_c",
        }
    )
    puts = work[work["option_type"] == "put"][put_cols].rename(
        columns={
            "mid": "mid_p",
            "bid": "bid_p",
            "ask": "ask_p",
        }
    )
    pairs = calls.merge(puts, on=["expiry", "strike"], how="inner")
    if pairs.empty:
        return 0, pairs
    pairs["parity_ok"] = pd.Series(True, index=pairs.index, dtype=object)
    pairs["parity_severity"] = "ok"
    forwards = _normalise_forwards(forwards)
    has_quotes = {"bid_c", "ask_c", "bid_p", "ask_p"}.issubset(pairs.columns)

    for expiry, group in pairs.groupby("expiry"):
        params = _params_for_expiry(forwards, expiry)
        if params is None:
            pairs.loc[group.index, "parity_ok"] = np.nan
            pairs.loc[group.index, "parity_severity"] = "no_valid_forward"
            continue
        forward, discount = params
        target = discount * (forward - group["strike"].to_numpy(dtype=float))
        mid_diff = (
            group["mid_c"].to_numpy(dtype=float)
            - group["mid_p"].to_numpy(dtype=float)
        )
        ok, severity, _ = _parity_ok_and_severity(
            target,
            mid_diff,
            group["bid_c"].to_numpy(dtype=float) if has_quotes else None,
            group["ask_c"].to_numpy(dtype=float) if has_quotes else None,
            group["bid_p"].to_numpy(dtype=float) if has_quotes else None,
            group["ask_p"].to_numpy(dtype=float) if has_quotes else None,
            abs_tol,
        )
        pairs.loc[group.index, "parity_ok"] = ok
        pairs.loc[group.index, "parity_severity"] = severity

    filled = pairs["parity_severity"].fillna("")
    violations = int((filled == "hard").sum() + (filled == "mid_only").sum())
    return violations, pairs


def check_monotonicity(
    df: pd.DataFrame,
    good_only: bool = True,
    tol: float = 1e-4,
) -> tuple[int, pd.DataFrame]:
    """Cross-strike monotonicity: calls fall, puts rise with strike."""
    work = _quality_frame(df, good_only)
    work["mono_ok"] = True
    violations = 0
    for (expiry, option_type), group in work.groupby(
        ["expiry", "option_type"]
    ):
        ordered = group.sort_values("strike")
        if len(ordered) < 2:
            continue
        diffs = np.diff(ordered["mid"].to_numpy(dtype=float))
        bad = (
            diffs > tol
            if option_type == "call"
            else diffs < -tol
        )
        violations += int(bad.sum())
        if bad.any():
            bad_index = ordered.index[1:][bad]
            work.loc[bad_index, "mono_ok"] = False
    return violations, work


def check_butterfly(
    df: pd.DataFrame,
    good_only: bool = True,
    tol: float = 1e-4,
) -> tuple[int, pd.DataFrame]:
    """Slope-form butterfly convexity valid for uneven strike spacing."""
    work = _quality_frame(df, good_only)
    work["butterfly_ok"] = True
    violations = 0
    for (expiry, option_type), group in work.groupby(
        ["expiry", "option_type"]
    ):
        ordered = group.sort_values("strike")
        if len(ordered) < 3:
            continue
        strikes = ordered["strike"].to_numpy(dtype=float)
        prices = ordered["mid"].to_numpy(dtype=float)
        for i in range(1, len(strikes) - 1):
            left_gap = strikes[i] - strikes[i - 1]
            right_gap = strikes[i + 1] - strikes[i]
            if left_gap <= 0 or right_gap <= 0:
                continue
            slope_left = (prices[i] - prices[i - 1]) / left_gap
            slope_right = (prices[i + 1] - prices[i]) / right_gap
            if slope_right < slope_left - tol:
                work.loc[ordered.index[i], "butterfly_ok"] = False
                violations += 1
    return violations, work


def check_calendar(
    df: pd.DataFrame,
    good_only: bool = True,
    tol: float = 1e-4,
) -> tuple[int, pd.DataFrame]:
    """Same-strike price monotonicity in expiry (diagnostic for dividends)."""
    work = _quality_frame(df, good_only)
    work["calendar_ok"] = True
    violations = 0
    for (strike, option_type), group in work.groupby(
        ["strike", "option_type"]
    ):
        ordered = group.sort_values("time_to_expiry")
        if len(ordered) < 2:
            continue
        diffs = np.diff(ordered["mid"].to_numpy(dtype=float))
        bad = diffs < -tol
        violations += int(bad.sum())
        if bad.any():
            bad_index = ordered.index[1:][bad]
            work.loc[bad_index, "calendar_ok"] = False
    return violations, work


def run_all_checks(
    df: pd.DataFrame,
    forwards: pd.DataFrame,
    good_only: bool = True,
) -> ArbitrageReport:
    """Run the five checks and assemble the report with detail frames."""
    bounds_df = check_bounds(df, forwards, good_only=good_only)
    parity_viol, parity_df = check_parity(df, forwards, good_only=good_only)
    mono_viol, mono_df = check_monotonicity(df, good_only=good_only)
    butterfly_viol, butterfly_df = check_butterfly(
        df,
        good_only=good_only,
    )
    calendar_viol, calendar_df = check_calendar(
        df,
        good_only=good_only,
    )

    severity = bounds_df["bound_severity"].fillna("")
    bound_viol = int(
        ((severity == "hard") | (severity == "mid_only")).sum()
    )
    if good_only:
        total = int((bounds_df["quality"] == GOOD_QUALITY).sum())
    else:
        total = int(len(bounds_df))

    return ArbitrageReport(
        bound_violations=bound_viol,
        parity_violations=parity_viol,
        monotonicity_violations=mono_viol,
        butterfly_violations=butterfly_viol,
        calendar_violations=calendar_viol,
        total_checked=total,
        notes=[
            "bound/parity hard = tradable at quotes; mid_only = mid outside "
            "but quotes not (diagnostic)",
            "calendar checks are diagnostics for dividend-paying SPY: longer "
            "maturity need not be more expensive across an ex-dividend date",
            "F/D are fitted from the same mids, so bound/parity counts partly "
            "reflect regression residuals rather than tradable profit",
        ],
        details={
            "bounds": bounds_df,
            "parity": parity_df,
            "monotonicity": mono_df,
            "butterfly": butterfly_df,
            "calendar": calendar_df,
        },
    )
