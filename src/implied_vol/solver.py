"""Black-76 implied-volatility solver with Newton + Brent fallback.

Black-76 is parameterised by the implied forward ``F`` and discount factor
``D`` from ``src.implied_vol.forward``, so no separate risk-free rate or
dividend yield is needed inside the solver.

Design rules inherited from the option-A roadmap and its review:

- prices are validated against the European F/D no-arbitrage bounds before
  solving; below-intrinsic prices return ``at_intrinsic`` (sigma ~ 0) and
  out-of-bound prices return ``no_arb`` with NaN;
- Newton-Raphson uses a Brenner-Subrahmanyam initial guess and switches to
  Brent when vega is tiny or sigma leaves the valid band;
- mid quotes are unified to the OTM side through put-call parity so the
  skew is clean, while bid/ask implied vols always use the contract's own
  quotes (parity only holds at the mid level);
- every failure mode is explicit through ``status``; iterations guard
  against NaN/Inf drift instead of relying on the fallback to catch it.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from .forward import lookup_forward_params


MIN_VOL = 1e-4
MAX_VOL = 5.0
MIN_TTM = 1e-6
VEGA_FLOOR = 1e-8
PRICE_TOL = 1e-8
INTRINSIC_TOL = 1e-6


def _intrinsic(forward: float, strike: float, is_call: bool) -> float:
    return (
        max(forward - strike, 0.0)
        if is_call
        else max(strike - forward, 0.0)
    )


def black76_price(
    forward: float,
    strike: float,
    time_to_expiry: float,
    sigma: float,
    discount: float,
    is_call: bool,
) -> float:
    """Black-76 option price with sigma/T boundary handling."""
    if not math.isfinite(forward) or not math.isfinite(strike):
        raise ValueError("forward and strike must be finite.")
    if forward <= 0 or strike <= 0:
        raise ValueError("forward and strike must be positive.")
    if not math.isfinite(discount):
        raise ValueError("discount must be finite.")
    if time_to_expiry < 0 or not math.isfinite(time_to_expiry):
        raise ValueError("time_to_expiry must be non-negative and finite.")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative.")
    if time_to_expiry <= MIN_TTM or sigma <= 0:
        return discount * _intrinsic(forward, strike, is_call)

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(forward / strike) + 0.5 * sigma**2 * time_to_expiry
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if is_call:
        price = discount * (
            forward * norm.cdf(d1) - strike * norm.cdf(d2)
        )
    else:
        price = discount * (
            strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
        )
    return float(price)


def black76_vega(
    forward: float,
    strike: float,
    time_to_expiry: float,
    sigma: float,
    discount: float,
) -> float:
    """Black-76 vega (per 1.00 volatility unit)."""
    if (
        time_to_expiry <= MIN_TTM
        or sigma <= 0
        or forward <= 0
        or strike <= 0
    ):
        return 0.0
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(forward / strike) + 0.5 * sigma**2 * time_to_expiry
    ) / (sigma * sqrt_t)
    return float(discount * forward * norm.pdf(d1) * sqrt_t)


def _price_bounds(
    forward: float,
    strike: float,
    discount: float,
    is_call: bool,
) -> tuple[float, float]:
    lower = discount * _intrinsic(forward, strike, is_call)
    upper = discount * (
        forward if is_call else strike
    )
    return lower, upper


def _initial_guess(
    price: float,
    forward: float,
    discount: float,
    time_to_expiry: float,
) -> float:
    """Brenner-Subrahmanyam ATM approximation, clipped to a sane band."""
    if time_to_expiry <= MIN_TTM:
        return 0.2
    guess = (
        math.sqrt(2.0 * math.pi / time_to_expiry)
        * price
        / (discount * forward)
    )
    return float(np.clip(guess, 0.05, 2.0))


def implied_vol(
    price: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    discount: float,
    is_call: bool,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> tuple[float, str]:
    """Solve one implied volatility; returns ``(iv, status)``."""
    if not math.isfinite(price) or price < 0:
        return float("nan"), "no_arb"
    if not (
        math.isfinite(forward)
        and math.isfinite(strike)
        and math.isfinite(discount)
    ):
        return float("nan"), "no_arb"
    if forward <= 0 or strike <= 0 or discount <= 0:
        return float("nan"), "no_arb"
    if time_to_expiry <= MIN_TTM:
        return float("nan"), "expired"

    lower, upper = _price_bounds(
        forward,
        strike,
        discount,
        is_call,
    )
    if price <= lower + INTRINSIC_TOL:
        if price < lower - PRICE_TOL:
            return float("nan"), "no_arb"
        return MIN_VOL, "at_intrinsic"
    if price > upper + PRICE_TOL:
        return float("nan"), "no_arb"

    sigma = _initial_guess(price, forward, discount, time_to_expiry)
    for _ in range(max_iter):
        model = black76_price(
            forward,
            strike,
            time_to_expiry,
            sigma,
            discount,
            is_call,
        )
        if abs(model - price) < tol:
            return float(sigma), "ok"
        vega = black76_vega(
            forward,
            strike,
            time_to_expiry,
            sigma,
            discount,
        )
        if not np.isfinite(vega) or vega < VEGA_FLOOR:
            break
        sigma = sigma - (model - price) / vega
        if (
            not np.isfinite(sigma)
            or sigma <= MIN_VOL
            or sigma >= MAX_VOL
        ):
            break

    def objective(vol: float) -> float:
        return black76_price(
            forward,
            strike,
            time_to_expiry,
            vol,
            discount,
            is_call,
        ) - price

    f_low = objective(MIN_VOL)
    f_high = objective(MAX_VOL)
    if not (np.isfinite(f_low) and np.isfinite(f_high)):
        return float("nan"), "converge_fail"
    if f_low * f_high > 0:
        return float("nan"), "converge_fail"
    try:
        iv = brentq(
            objective,
            MIN_VOL,
            MAX_VOL,
            xtol=tol,
            maxiter=100,
        )
    except (ValueError, RuntimeError):
        return float("nan"), "converge_fail"
    return float(iv), "ok"


def _otm_side(forward: float, strike: float) -> str:
    """Preferred OTM side: call when strike >= forward, else put."""
    return "call" if strike >= forward else "put"


def solve_chain_iv(
    df: pd.DataFrame,
    forwards: pd.DataFrame,
    good_only: bool = True,
) -> pd.DataFrame:
    """Solve bid/mid/ask implied vols on a chain.

    bid/ask use the contract's own quotes; mid is unified to the OTM side.
    """
    work = df.copy()
    if good_only:
        if "quality" not in work.columns:
            raise ValueError(
                "good_only=True requires a 'quality' column."
            )
        work = work[work["quality"] == "good"]
    for column in ("iv_bid", "iv_mid", "iv_ask"):
        work[column] = float("nan")
    work["iv_status"] = ""
    work["iv_source_type"] = ""

    for expiry, group in work.groupby("expiry"):
        params = lookup_forward_params(forwards, expiry)
        if params is None:
            work.loc[group.index, "iv_status"] = "no_forward"
            continue
        forward, discount = params
        time_to_expiry = group["time_to_expiry"].astype(float)
        if (time_to_expiry <= 0).any():
            work.loc[
                group.index[time_to_expiry <= 0],
                "iv_status",
            ] = "expired"

        for index, row in group.iterrows():
            strike = float(row["strike"])
            is_call = bool(row["option_type"] == "call")
            t = float(row["time_to_expiry"])
            if t <= 0:
                continue

            own_prices = {
                "iv_bid": ("bid", is_call),
                "iv_ask": ("ask", is_call),
            }
            for iv_col, (price_col, call_flag) in own_prices.items():
                price = row[price_col]
                if (
                    price_col not in row.index
                    or not np.isfinite(float(price))
                    or float(price) <= 0
                ):
                    continue
                iv, _ = implied_vol(
                    float(price),
                    forward,
                    strike,
                    t,
                    discount,
                    call_flag,
                )
                work.loc[index, iv_col] = float(iv)

            # Mid: unify to the OTM side via put-call parity.
            side = _otm_side(forward, strike)
            source = group[
                (group["strike"] == strike)
                & (group["option_type"] == side)
            ]
            if not source.empty:
                solve_price = float(source.iloc[0]["mid"])
                solve_call = side == "call"
            else:
                # Preferred side missing: synthesise it from the counterpart
                # mid through parity instead of solving the wrong side.
                other_side = "put" if side == "call" else "call"
                other = group[
                    (group["strike"] == strike)
                    & (group["option_type"] == other_side)
                ]
                if other.empty:
                    work.loc[index, "iv_status"] = "no_otm_counterpart"
                    continue
                other_mid = float(other.iloc[0]["mid"])
                if side == "call":
                    solve_price = other_mid + discount * (
                        forward - strike
                    )
                    solve_call = True
                else:
                    solve_price = other_mid - discount * (
                        forward - strike
                    )
                    solve_call = False
            iv, status = implied_vol(
                solve_price,
                forward,
                strike,
                t,
                discount,
                solve_call,
            )
            work.loc[index, "iv_mid"] = float(iv)
            work.loc[index, "iv_status"] = status
            work.loc[index, "iv_source_type"] = side
    return work
