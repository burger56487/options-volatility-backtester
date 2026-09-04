"""VaR backtesting: Kupiec and Christoffersen tests."""

from __future__ import annotations

import math

import numpy as np


def _lr_ratio(hits, n, alpha):
    if n <= 0:
        return float("nan")
    observed = hits / n
    if observed <= 0 or observed >= 1:
        return float("nan")
    likelihood = -2.0 * (
        hits * math.log(alpha / observed)
        + (n - hits) * math.log((1.0 - alpha) / (1.0 - observed))
    )
    return max(likelihood, 0.0)


def kupiec_unconditional_coverage(
    pnl: np.ndarray,
    var_thresholds: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Kupiec unconditional-coverage test."""
    pnl = np.asarray(pnl, dtype=float)
    var_thresholds = np.asarray(var_thresholds, dtype=float)
    n = int(min(len(pnl), len(var_thresholds)))
    if n < 5:
        return {"n": n, "p_value": float("nan")}
    hits = int(np.sum(-pnl[:n] > var_thresholds[:n]))
    lr = _lr_ratio(hits, n, alpha)
    p_value = float(np.nan)
    if not math.isnan(lr):
        from scipy.stats import chi2

        p_value = float(1.0 - chi2.cdf(lr, df=1))
    return {"n": n, "exceptions": hits, "expected": alpha * n, "p_value": p_value}


def christoffersen_independence(
    pnl: np.ndarray,
    var_thresholds: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Christoffersen independence test via a two-state Markov chain."""
    pnl = np.asarray(pnl, dtype=float)
    var_thresholds = np.asarray(var_thresholds, dtype=float)
    n = int(min(len(pnl), len(var_thresholds)))
    if n < 5:
        return {"n": n, "p_value": float("nan")}
    hits = (-pnl[:n] > var_thresholds[:n]).astype(int)
    transitions = {"00": 0, "01": 0, "10": 0, "11": 0}
    for i in range(1, n):
        transitions[f"{hits[i-1]}{hits[i]}"] += 1
    total_after_zero = transitions["00"] + transitions["01"]
    total_after_one = transitions["10"] + transitions["11"]
    pi01 = (
        transitions["01"] / total_after_zero
        if total_after_zero
        else 0.5
    )
    pi11 = (
        transitions["11"] / total_after_one
        if total_after_one
        else 0.5
    )
    pi = hits.mean()
    log_likelihood_alt = 0.0
    for key, count in transitions.items():
        prob = pi01 if key == "01" else pi11 if key == "11" else 1.0 - (
            pi01 if key.startswith("0") else pi11
        )
        prob = min(max(prob, 1e-12), 1.0 - 1e-12)
        log_likelihood_alt += count * math.log(prob)
    log_likelihood_null = 0.0
    for key, count in transitions.items():
        prob = 1.0 - pi if key.endswith("0") else pi
        prob = min(max(prob, 1e-12), 1.0 - 1e-12)
        log_likelihood_null += count * math.log(prob)
    lr = -2.0 * (log_likelihood_null - log_likelihood_alt)
    from scipy.stats import chi2

    p_value = float(1.0 - chi2.cdf(max(lr, 0.0), df=1))
    return {
        "n": n,
        "transitions": transitions,
        "p_value": p_value,
    }
