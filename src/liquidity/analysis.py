"""Chain-liquidity analysis (stage 9, formal version).

Design decisions:

- the module consumes the *full* snapshot and deliberately does NOT filter
  on quality == "good": wide-spread/illiquid contracts are themselves the
  evidence of poor liquidity, so filtering would hide the full picture
  (this is the opposite of skew/Greeks, which use the good subset);
- the reliability thresholds are heuristic and are exposed as module
  constants so they can be re-tuned; a sensitivity analysis is recommended
  before production use;
- vectorised ``rate_reliability`` and single-row ``_reliability_score`` are
  kept in sync and must agree exactly (regression-tested);
- when ``iv_bid``/``iv_ask`` are absent, the IV bid/ask band is computed
  through the repository's Black-Scholes implied-vol solver so liquidity
  state can still reflect IV uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


MIN_MID = 1e-4
MIN_OI = 10
MIN_VOLUME = 1
IV_SPREAD_WIDE = 0.05


def _numeric_series(series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(
        dtype=float
    )


def compute_spread_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Absolute/relative spread with division protection."""
    work = df.copy()
    if "abs_spread" not in work.columns:
        if {"bid", "ask"}.issubset(work.columns):
            work["abs_spread"] = (
                pd.to_numeric(work["ask"], errors="coerce")
                - pd.to_numeric(work["bid"], errors="coerce")
            )
        else:
            work["abs_spread"] = np.nan
    if "mid" not in work.columns:
        work["mid"] = 0.5 * (
            pd.to_numeric(work.get("bid"), errors="coerce")
            + pd.to_numeric(work.get("ask"), errors="coerce")
        )
    mid = pd.to_numeric(work["mid"], errors="coerce")
    safe_mid = mid.clip(lower=MIN_MID)
    work["rel_spread"] = work["abs_spread"] / safe_mid
    work.loc[mid < MIN_MID, "rel_spread"] = np.nan
    return work


def compute_iv_uncertainty(
    df: pd.DataFrame,
    *,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.012,
) -> pd.DataFrame:
    """IV bid/ask band width; solves IV when the columns are missing."""
    work = df.copy()
    has_band = {"iv_ask", "iv_bid"}.issubset(work.columns)
    if not has_band and {
        "bid",
        "ask",
        "strike",
        "spot",
        "time_to_expiry",
        "option_type",
    }.issubset(work.columns):
        try:
            from src.market_data.real_option_chain import add_iv_band

            solved = add_iv_band(
                work,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )
            work["iv_bid"] = solved["iv_bid"]
            work["iv_ask"] = solved["iv_ask"]
            has_band = True
        except Exception:  # noqa: BLE001 - fall back to NaN below
            has_band = False
    if has_band:
        work["iv_spread"] = (
            pd.to_numeric(work["iv_ask"], errors="coerce")
            - pd.to_numeric(work["iv_bid"], errors="coerce")
        )
        work.loc[work["iv_spread"] < 0, "iv_spread"] = np.nan
    else:
        work["iv_spread"] = np.nan
    return work


def _rel_spread_penalty(rs: float) -> float:
    if not np.isfinite(rs):
        return 30.0
    if rs > 0.3:
        return 40.0
    if rs > 0.1:
        return 20.0
    return 0.0


def _oi_penalty(oi: float) -> float:
    if oi < MIN_OI:
        return 30.0
    if oi < 100:
        return 10.0
    return 0.0


def _volume_penalty(volume: float) -> float:
    return 15.0 if volume < MIN_VOLUME else 0.0


def _iv_spread_penalty(iv_spread: float) -> float:
    if not np.isfinite(iv_spread):
        return 15.0
    if iv_spread > IV_SPREAD_WIDE:
        return 20.0
    return 0.0


def rate_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-factor reliability score (0-100) and low/medium/high label."""
    work = df.copy()
    if "rel_spread" not in work.columns:
        work = compute_spread_metrics(work)
    if "iv_spread" not in work.columns:
        work = compute_iv_uncertainty(work)

    rs = pd.to_numeric(
        work.get("rel_spread"),
        errors="coerce",
    ).to_numpy(dtype=float)
    oi = _numeric_series(work.get("open_interest", pd.Series(0.0)))
    volume = _numeric_series(work.get("volume", pd.Series(0.0)))
    iv_spread = pd.to_numeric(
        work.get("iv_spread"),
        errors="coerce",
    ).to_numpy(dtype=float)

    rs_penalty = np.where(
        np.isnan(rs),
        30.0,
        np.where(rs > 0.3, 40.0, np.where(rs > 0.1, 20.0, 0.0)),
    )
    oi_penalty = np.where(oi < MIN_OI, 30.0, np.where(oi < 100, 10.0, 0.0))
    volume_penalty = np.where(volume < MIN_VOLUME, 15.0, 0.0)
    iv_penalty = np.where(
        np.isnan(iv_spread),
        15.0,
        np.where(iv_spread > IV_SPREAD_WIDE, 20.0, 0.0),
    )
    score = np.clip(
        100.0 - rs_penalty - oi_penalty - volume_penalty - iv_penalty,
        0.0,
        100.0,
    )
    work["reliability_score"] = score
    work["reliability"] = pd.cut(
        score,
        bins=[-1.0, 40.0, 70.0, 101.0],
        labels=["low", "medium", "high"],
    )
    return work


def _reliability_score(row) -> float:
    """Single-row reliability score; must match ``rate_reliability``."""
    rs_value = row.get("rel_spread", np.nan)
    rs = (
        float(rs_value)
        if rs_value is not None and not pd.isna(rs_value)
        else np.nan
    )
    oi_value = row.get("open_interest", 0.0)
    oi = (
        0.0
        if oi_value is None or pd.isna(oi_value)
        else float(oi_value)
    )
    volume_value = row.get("volume", 0.0)
    volume = (
        0.0
        if volume_value is None or pd.isna(volume_value)
        else float(volume_value)
    )
    iv_value = row.get("iv_spread", np.nan)
    iv_spread = (
        float(iv_value)
        if iv_value is not None and not pd.isna(iv_value)
        else np.nan
    )
    score = (
        100.0
        - _rel_spread_penalty(rs)
        - _oi_penalty(oi)
        - _volume_penalty(volume)
        - _iv_spread_penalty(iv_spread)
    )
    return max(0.0, float(score))


def _with_moneyness(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "moneyness" not in work.columns:
        if {"strike", "spot"}.issubset(work.columns):
            work["moneyness"] = (
                pd.to_numeric(work["strike"], errors="coerce")
                / pd.to_numeric(work["spot"], errors="coerce")
            )
        else:
            work["moneyness"] = np.nan
    return work


def liquidity_by_moneyness(df: pd.DataFrame, n_bins: int = 7) -> pd.DataFrame:
    """Relative spread vs moneyness in adaptive quantile bins."""
    work = _with_moneyness(df)
    if "rel_spread" not in work.columns:
        work = compute_spread_metrics(work)
    work = work.dropna(subset=["rel_spread", "moneyness"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["log_moneyness"] = np.log(
        pd.to_numeric(work["moneyness"], errors="coerce")
    )
    work = work.dropna(subset=["log_moneyness"])
    if work.empty:
        return pd.DataFrame()
    try:
        work["lm_bucket"] = pd.qcut(
            work["log_moneyness"],
            q=n_bins,
            duplicates="drop",
        )
    except ValueError:
        work["lm_bucket"] = pd.cut(
            work["log_moneyness"],
            bins=n_bins,
        )
    return (
        work.groupby("lm_bucket", observed=True)
        .agg(
            mean_rel_spread=("rel_spread", "mean"),
            median_rel_spread=("rel_spread", "median"),
            count=("rel_spread", "size"),
        )
        .reset_index()
    )


def liquidity_by_expiry(df: pd.DataFrame) -> pd.DataFrame:
    """Relative spread and traded volume vs expiry."""
    work = compute_spread_metrics(df)
    work = work.dropna(subset=["rel_spread", "expiry"]).copy()
    if work.empty:
        return pd.DataFrame()
    for column in ("open_interest", "volume"):
        if column not in work.columns:
            work[column] = 0.0
    return (
        work.groupby("expiry", observed=True)
        .agg(
            mean_rel_spread=("rel_spread", "mean"),
            mean_abs_spread=("abs_spread", "mean"),
            total_oi=("open_interest", "sum"),
            total_volume=("volume", "sum"),
            count=("rel_spread", "size"),
        )
        .reset_index()
    )


@dataclass
class LiquidityState:
    overall_state: str  # liquid / moderate / illiquid / unknown
    median_rel_spread: float
    pct_reliable: float
    pct_good: float
    total_open_interest: float
    total_volume: float
    by_moneyness: pd.DataFrame | None = None
    by_expiry: pd.DataFrame | None = None
    warnings: list[str] = field(default_factory=list)


def assess_liquidity_state(df: pd.DataFrame) -> LiquidityState:
    """Overall liquidity state from the full snapshot."""
    work = compute_spread_metrics(df)
    work = compute_iv_uncertainty(work)
    work = rate_reliability(work)

    valid = work.dropna(subset=["rel_spread"])
    if valid.empty:
        return LiquidityState(
            overall_state="unknown",
            median_rel_spread=float("nan"),
            pct_reliable=0.0,
            pct_good=0.0,
            total_open_interest=0.0,
            total_volume=0.0,
            warnings=["无有效价差数据"],
        )

    median_rs = float(valid["rel_spread"].median())
    pct_high = float(
        (work["reliability"] == "high").sum() / len(work) * 100.0
    )
    pct_good = 100.0
    if "quality" in work.columns:
        pct_good = 100.0 * float((work["quality"] == "good").mean())

    total_oi = float(_numeric_series(work.get("open_interest", pd.Series(0.0))).sum())
    total_volume = float(_numeric_series(work.get("volume", pd.Series(0.0))).sum())

    warnings: list[str] = []
    if median_rs < 0.05 and pct_high > 50:
        state = "liquid"
    elif median_rs < 0.15 and pct_high > 25:
        state = "moderate"
    else:
        state = "illiquid"
        warnings.append("整体流动性较差，隐含波动率可靠性低")

    by_expiry = (
        liquidity_by_expiry(work)
        if "expiry" in work.columns
        else None
    )
    return LiquidityState(
        overall_state=state,
        median_rel_spread=median_rs,
        pct_reliable=pct_high,
        pct_good=pct_good,
        total_open_interest=total_oi,
        total_volume=total_volume,
        by_moneyness=liquidity_by_moneyness(work),
        by_expiry=by_expiry,
        warnings=warnings,
    )
