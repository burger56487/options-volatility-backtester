"""Hedge evaluation metrics and a fair-comparison simulator."""

from __future__ import annotations

import math

import numpy as np

from .policies import FixedDeltaHedge, HedgeState, ThresholdDeltaHedge


def simulate_hedge(
    spot_path,
    option_delta_path,
    policy,
    cost_bps: float = 1.0,
) -> dict:
    """Simulate one path of hedging and return error/cost metrics.

    Daily PnL = -shares_prev * dS (hedge leg) plus the unhedged option move is
    approximated by -delta*dS; the reported hedge error is the residual after
    accounting for the delta move and transaction costs.
    """
    spot = np.asarray(spot_path, dtype=float)
    deltas = np.asarray(option_delta_path, dtype=float)
    shares = 0.0
    daily_pnl = []
    costs = []
    for i in range(1, len(spot)):
        d_spot = spot[i] - spot[i - 1]
        state = HedgeState(
            spot=float(spot[i - 1]),
            delta=float(deltas[i - 1]),
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            current_shares=shares,
        )
        target = policy.target_shares(state)
        trade = target - shares
        cost = abs(trade) * spot[i - 1] * cost_bps / 10_000.0
        # Hedge PnL from the move plus option delta move (delta-neutralised).
        hedge_pnl = -shares * d_spot
        residual = hedge_pnl - (-deltas[i - 1]) * d_spot
        daily_pnl.append(residual - cost)
        costs.append(cost)
        shares = target
    pnl = np.asarray(daily_pnl)
    return {
        "hedge_error_std": float(np.std(pnl, ddof=1))
        if len(pnl) > 1
        else 0.0,
        "total_cost": float(sum(costs)),
        "trade_count": int(sum(1 for c in costs if c > 0)),
        "terminal_pnl": float(np.sum(pnl)),
    }


def compare_policies(spot_path, option_delta_path, cost_bps=1.0) -> dict:
    """Fair comparison of fixed vs threshold delta hedging on one path."""
    return {
        "fixed": simulate_hedge(
            spot_path,
            option_delta_path,
            FixedDeltaHedge(),
            cost_bps,
        ),
        "threshold": simulate_hedge(
            spot_path,
            option_delta_path,
            ThresholdDeltaHedge(band_shares=5.0),
            cost_bps,
        ),
    }
