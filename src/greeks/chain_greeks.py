"""Chain-level Greeks, portfolio aggregation and risk heatmaps.

The per-contract formulas reuse ``src.pricing.black_scholes.price_and_greeks``
(spot-delta convention, vega per 1.00 volatility unit, theta per year, rho
per 1.00 rate unit) so the chain module cannot drift from the pricing
engine.  Raw and converted units are kept as separate columns:

- ``delta`` / ``gamma`` / ``rho``: raw units;
- ``vega``: per 1.00 (100%) volatility change; ``vega_per_pct`` = vega/100;
- ``theta``: per year; ``theta_per_day`` = theta/365.

Rate convention: the real SPY chain used in this repository has 4-11 day
expiries where the regression-based implied rate/dividend split is unstable
(see report 3.1.2), so the chain Greeks default to the repository's pricing
convention r=4%, q=1.2% and record the parameters actually used.

Heatmap caveat: gamma/vega are identical for call and put, so they can be
pooled per (strike, expiry); delta has opposite signs, so a delta heatmap
requires an explicit option_type filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pricing.black_scholes import price_and_greeks


GREEKS = ["delta", "gamma", "vega", "theta", "rho"]


def compute_greeks_row(
    spot: float,
    strike: float,
    time_to_expiry: float,
    sigma: float,
    risk_free_rate: float,
    dividend_yield: float,
    is_call: bool,
) -> dict[str, float]:
    """Full BS Greeks for one row; NaN on invalid inputs."""
    nan_result = {name: float("nan") for name in GREEKS}
    if (
        not np.isfinite(sigma)
        or sigma <= 0
        or not np.isfinite(time_to_expiry)
        or time_to_expiry <= 0
        or not np.isfinite(spot)
        or spot <= 0
        or not np.isfinite(strike)
        or strike <= 0
    ):
        return nan_result
    result = price_and_greeks(
        spot=float(spot),
        strike=float(strike),
        time_to_expiry=float(time_to_expiry),
        risk_free_rate=float(risk_free_rate),
        volatility=float(sigma),
        option_type="call" if is_call else "put",
        dividend_yield=float(dividend_yield),
    )
    return {
        "delta": float(result.delta),
        "gamma": float(result.gamma),
        "vega": float(result.vega),
        "theta": float(result.theta),
        "rho": float(result.rho),
    }


def compute_chain_greeks(
    df: pd.DataFrame,
    *,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.012,
    good_only: bool = True,
) -> pd.DataFrame:
    """Add raw and converted Greeks to every contract on the chain."""
    work = df.copy()
    if good_only:
        if "quality" not in work.columns:
            raise ValueError(
                "good_only=True requires a 'quality' column."
            )
        work = work[work["quality"] == "good"]
    required = {
        "spot",
        "strike",
        "time_to_expiry",
        "iv_mid",
        "option_type",
    }
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    work = work[
        work["iv_mid"].notna()
        & np.isfinite(work["iv_mid"].astype(float))
        & (work["iv_mid"].astype(float) > 0)
    ].copy()
    for name in GREEKS + ["vega_per_pct", "theta_per_day"]:
        work[name] = float("nan")
    work["_is_call"] = work["option_type"] == "call"

    for index, row in work.iterrows():
        greeks = compute_greeks_row(
            float(row["spot"]),
            float(row["strike"]),
            float(row["time_to_expiry"]),
            float(row["iv_mid"]),
            risk_free_rate,
            dividend_yield,
            bool(row["_is_call"]),
        )
        for name, value in greeks.items():
            work.loc[index, name] = value
    work["vega_per_pct"] = work["vega"] / 100.0
    work["theta_per_day"] = work["theta"] / 365.0
    work = work.drop(columns=["_is_call"])
    work.attrs["greeks_convention"] = {
        "risk_free_rate": risk_free_rate,
        "dividend_yield": dividend_yield,
        "delta": "spot delta (shares per contract)",
        "gamma": "delta change per 1.00 spot move",
        "vega": "per 1.00 volatility; vega_per_pct per 1%",
        "theta": "per year; theta_per_day per day",
        "rho": "per 1.00 rate change",
    }
    return work


def aggregate_portfolio_greeks(
    df: pd.DataFrame,
    *,
    position_column: str | None = None,
    multiplier: float = 100.0,
) -> dict[str, float]:
    """Aggregate Greeks assuming one contract per row unless positions exist.

    Aggregation uses the raw-unit columns; vega_per_pct and theta_per_day are
    then derived once from the raw sums so units cannot be double-converted.
    """
    work = df.copy()
    if position_column is None:
        work["_position"] = 1.0
    else:
        if position_column not in work.columns:
            raise ValueError(
                f"position column missing: {position_column}"
            )
        work["_position"] = pd.to_numeric(
            work[position_column],
            errors="coerce",
        ).fillna(0.0)
    weight = work["_position"].to_numpy(dtype=float) * multiplier
    output = {}
    for name in GREEKS:
        values = pd.to_numeric(work[name], errors="coerce").to_numpy(
            dtype=float
        )
        output[name] = float(np.nansum(values * weight))
    output["vega_per_pct"] = output["vega"] / 100.0
    output["theta_per_day"] = output["theta"] / 365.0
    return output


def build_greek_heatmap(
    df: pd.DataFrame,
    greek: str,
    *,
    option_type: str | None = None,
) -> pd.DataFrame:
    """Pivot strike x time-to-expiry; delta requires an option type."""
    if greek not in GREEKS:
        raise ValueError(f"Unknown greek: {greek}")
    work = df[df[greek].notna()].copy()
    if greek == "delta" and option_type is None:
        raise ValueError(
            "delta heatmaps require option_type='call' or 'put' "
            "(delta signs differ by type)."
        )
    if option_type is not None:
        work = work[work["option_type"] == option_type]
    pivot = work.pivot_table(
        index="strike",
        columns="time_to_expiry",
        values=greek,
        aggfunc="mean",
    )
    pivot = pivot.sort_index().sort_index(axis=1)
    return pivot


def top_risk_contracts(
    df: pd.DataFrame,
    greek: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Rows with the largest absolute exposure for one greek."""
    if greek not in GREEKS:
        raise ValueError(f"Unknown greek: {greek}")
    work = df[df[greek].notna()].copy()
    work["abs_greek"] = work[greek].abs()
    columns = [
        column
        for column in [
            "expiry",
            "strike",
            "option_type",
            "iv_mid",
            greek,
        ]
        if column in work.columns
    ]
    return work.nlargest(top_n, "abs_greek")[columns]
