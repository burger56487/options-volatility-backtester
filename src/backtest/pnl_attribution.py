"""Greeks-based daily P&L attribution for delta-hedged option trades.

The single-trade backtest records per-day spot, Greeks, hedge position and
hedge costs in its equity curve. This module decomposes each day's total P&L
change into Delta, Gamma, Vega, Theta, Rho and transaction-cost contributions
using a Taylor expansion around the previous day's exposures, and reports the
unexplained residual.
"""

from __future__ import annotations

import pandas as pd


_REQUIRED_COLUMNS = [
    "spot",
    "option_delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "hedge_position",
    "hedge_transaction_cost",
    "call_implied_volatility",
    "put_implied_volatility",
    "total_pnl",
]

_COMPONENT_COLUMNS = [
    "delta_contribution",
    "gamma_contribution",
    "vega_contribution",
    "theta_contribution",
    "rho_contribution",
    "cost_contribution",
]


def attribute_daily_pnl(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Attribute daily total P&L changes to risk-factor contributions.

    Contributions are evaluated with the previous day's exposures:

    - Delta: ``(option_delta + hedge_position)_{t-1} * dS``
    - Gamma: ``0.5 * gamma_{t-1} * dS^2``
    - Vega: ``vega_{t-1} * d(avg implied vol)``
    - Theta: ``theta_{t-1} * dt`` (dt in years)
    - Rho: zero in this simulator (the risk-free rate is constant)
    - Costs: negative of the day's hedge transaction cost
    """
    missing = [
        column
        for column in _REQUIRED_COLUMNS
        if column not in equity_curve.columns
    ]
    if missing:
        raise ValueError(
            f"equity_curve missing required columns: {missing}"
        )

    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        raise TypeError(
            "equity_curve index must be a DatetimeIndex."
        )

    if len(equity_curve) < 2:
        raise ValueError(
            "at least two valuation rows are required for attribution."
        )

    df = equity_curve.copy()
    previous = df.shift(1)

    spot_change = df["spot"] - previous["spot"]
    day_fraction = (
        df.index.to_series().diff().dt.days / 365.0
    )

    implied_vol_avg = 0.5 * (
        df["call_implied_volatility"]
        + df["put_implied_volatility"]
    )
    implied_vol_avg_previous = 0.5 * (
        previous["call_implied_volatility"]
        + previous["put_implied_volatility"]
    )
    vol_change = implied_vol_avg - implied_vol_avg_previous

    delta_exposure = (
        previous["option_delta"] + previous["hedge_position"]
    )

    delta_contribution = delta_exposure * spot_change
    gamma_contribution = 0.5 * previous["gamma"] * spot_change**2
    vega_contribution = previous["vega"] * vol_change
    theta_contribution = previous["theta"] * day_fraction
    rho_contribution = pd.Series(0.0, index=df.index)
    cost_contribution = -df["hedge_transaction_cost"].fillna(0.0)

    actual_pnl_change = df["total_pnl"] - previous["total_pnl"]

    model_contribution = (
        delta_contribution.fillna(0.0)
        + gamma_contribution.fillna(0.0)
        + vega_contribution.fillna(0.0)
        + theta_contribution.fillna(0.0)
        + rho_contribution
        + cost_contribution
    )

    residual = actual_pnl_change - model_contribution

    return pd.DataFrame(
        {
            "actual_pnl_change": actual_pnl_change,
            "delta_contribution": delta_contribution,
            "gamma_contribution": gamma_contribution,
            "vega_contribution": vega_contribution,
            "theta_contribution": theta_contribution,
            "rho_contribution": rho_contribution,
            "cost_contribution": cost_contribution,
            "model_contribution": model_contribution,
            "residual": residual,
        }
    )


def attribution_summary(attribution: pd.DataFrame) -> dict[str, float]:
    """Aggregate a daily attribution table over the whole trade."""
    valid = attribution.iloc[1:]
    if valid.empty:
        return {
            "actual_pnl_total": 0.0,
            "residual_total": 0.0,
            "abs_residual_ratio": 0.0,
        }

    totals = {
        f"{column.replace('_contribution', '')}_total": float(
            valid[column].sum(skipna=True)
        )
        for column in _COMPONENT_COLUMNS
    }
    actual_total = float(valid["actual_pnl_change"].sum(skipna=True))
    residual_total = float(valid["residual"].sum(skipna=True))
    abs_actual = float(valid["actual_pnl_change"].abs().sum(skipna=True))

    abs_residual_ratio = (
        abs(residual_total) / abs_actual if abs_actual != 0.0 else 0.0
    )

    return {
        **totals,
        "actual_pnl_total": actual_total,
        "residual_total": residual_total,
        "abs_residual_ratio": abs_residual_ratio,
    }
