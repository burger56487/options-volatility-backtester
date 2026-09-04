"""Deterministic model-validation suite with a machine-readable report.

Every check re-executes project code on fixed parameters or on committed
data files, so the same suite can run locally, in CI, or from
``scripts/run_validation_report.py`` without network access.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain.enums import ExerciseStyle, OptionType
from src.pricing.binomial import crr_price
from src.pricing.black_scholes import option_price
from src.pricing.finite_difference import finite_difference_price
from src.pricing.heston import heston_mc_price
from src.pricing.merton import merton_mc_price, merton_series_price
from src.pricing.monte_carlo import monte_carlo_price
from src.pricing.requests import PricingRequest
from src.risk.backtest import (
    christoffersen_independence,
    kupiec_unconditional_coverage,
)
from src.volatility_surface.calibration import calibrate_svi
from src.volatility_surface.svi import svi_total_variance

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _request(**overrides) -> PricingRequest:
    params = dict(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        volatility=0.25,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
    )
    params.update(overrides)
    return PricingRequest(**params)


def _bs_reference() -> float:
    return option_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        volatility=0.25,
        option_type="call",
        dividend_yield=0.01,
    )


def _svi_rmse_by_expiry() -> dict:
    path = PROJECT_ROOT / "outputs/real_option_chain/spy_option_chain_active.csv"
    frame = pd.read_csv(path, parse_dates=["expiry"])
    result: dict[str, dict] = {}
    for expiry, group in frame.groupby(frame["expiry"].dt.date):
        rows = group.dropna(subset=["iv"])
        rows = rows[rows["iv"] > 0]
        k = rows["log_moneyness"].to_numpy()
        w = (rows["iv"] ** 2 * rows["time_to_expiry"]).to_numpy()
        if len(k) < 5:
            result[str(expiry)] = {"n": int(len(k)), "rmse_total_variance": None}
            continue
        params = calibrate_svi(k, w, minimum_points=5).parameters
        model = svi_total_variance(
            k,
            params["a"],
            params["b"],
            params["rho"],
            params["m"],
            params["sigma"],
        )
        result[str(expiry)] = {
            "n": int(len(k)),
            "rmse_total_variance": float(np.sqrt(np.mean((model - w) ** 2))),
        }
    return result


def _var_backtest_on_spy() -> dict:
    frame = pd.read_csv(
        PROJECT_ROOT / "data/raw/spy_daily_adjusted.csv",
        parse_dates=["Date"],
    ).set_index("Date")
    returns = frame["Close"].pct_change().dropna()
    rolling_var = (
        -returns.rolling(60)
        .apply(lambda x: np.quantile(x, 0.05), raw=True)
        .shift(1)
    )
    window = returns.iloc[60:]
    valid = pd.concat([window, rolling_var.loc[window.index]], axis=1).dropna()
    kupiec = kupiec_unconditional_coverage(
        valid.iloc[:, 0].to_numpy(),
        valid.iloc[:, 1].to_numpy(),
    )
    christoffersen = christoffersen_independence(
        valid.iloc[:, 0].to_numpy(),
        valid.iloc[:, 1].to_numpy(),
    )
    return {
        "backtest_days": int(kupiec["n"]),
        "exceptions": int(kupiec["exceptions"]),
        "expected": float(kupiec["expected"]),
        "kupiec_p": float(kupiec["p_value"]),
        "christoffersen_p": float(christoffersen["p_value"]),
    }


def run_validation_suite() -> dict:
    """Run all deterministic model-validation checks and return a report."""
    checks: list[dict] = []

    def record(
        name: str,
        description: str,
        passed: bool,
        value,
        threshold: str,
    ) -> None:
        checks.append(
            {
                "name": name,
                "description": description,
                "status": "pass" if passed else "fail",
                "value": value,
                "threshold": threshold,
            }
        )

    reference = _bs_reference()

    # CRR binomial converges to Black-Scholes as steps double.
    errors = []
    for steps in (100, 200, 400, 800):
        tree = crr_price(_request(steps=steps))
        errors.append(abs(tree.price - reference))
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    record(
        "crr_convergence_to_bs",
        "CRR binomial price converges to Black-Scholes.",
        errors[-1] < 5e-3 and all(ratio > 1.5 for ratio in ratios),
        {"final_absolute_error": errors[-1], "halving_ratios": ratios},
        "final error < 5e-3 and each step-doubling ratio > 1.5",
    )

    # Crank-Nicolson finite differences converge to Black-Scholes.
    fd = finite_difference_price(
        _request(),
        theta=0.5,
        x_steps=800,
        time_steps=400,
    )
    fd_error = abs(fd.price - reference)
    record(
        "crank_nicolson_to_bs",
        "Crank-Nicolson price converges to Black-Scholes.",
        fd_error < 5e-2,
        {"absolute_error": fd_error},
        "absolute error < 5e-2 at 400 time steps",
    )

    # Monte Carlo standard error scales as 1/sqrt(N).
    se_values = {
        n_paths: monte_carlo_price(
            _request(),
            n_paths=n_paths,
            seed=7,
        )["standard_error"]
        for n_paths in (10_000, 40_000, 160_000)
    }
    se_ratios = [
        se_values[10_000] / se_values[40_000],
        se_values[40_000] / se_values[160_000],
    ]
    record(
        "monte_carlo_se_scaling",
        "Monte Carlo standard error halves when paths quadruple.",
        all(1.7 < ratio < 2.3 for ratio in se_ratios),
        {"standard_errors": se_values, "quadrupling_ratios": se_ratios},
        "each 4x path ratio between 1.7 and 2.3",
    )

    # Heston with near-zero vol-of-vol degenerates to Black-Scholes.
    heston = heston_mc_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        kappa=1.0,
        theta=0.04,
        sigma_v=1e-6,
        rho=0.0,
        v0=0.04,
        risk_free_rate=0.04,
        option_type=OptionType.CALL,
        dividend_yield=0.0,
        n_paths=40_000,
        n_steps=20,
        seed=3,
    )
    heston_reference = option_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        volatility=math.sqrt(0.04),
        option_type="call",
        dividend_yield=0.0,
    )
    heston_error = abs(heston["price"] - heston_reference)
    record(
        "heston_degenerates_to_bs",
        "Heston MC with zero vol-of-vol matches Black-Scholes.",
        heston_error < 3 * heston["standard_error"],
        {
            "absolute_error": heston_error,
            "standard_error": heston["standard_error"],
        },
        "error < 3 standard errors (40k paths, seed 3)",
    )

    # Merton series with zero jump intensity degenerates to Black-Scholes.
    merton_series = merton_series_price(
        100.0,
        100.0,
        0.5,
        0.2,
        0.04,
        OptionType.CALL,
        jump_intensity=0.0,
        jump_mean=0.0,
        jump_vol=0.0,
        dividend_yield=0.0,
    )
    merton_degenerate_error = abs(merton_series - heston_reference)
    record(
        "merton_degenerates_to_bs",
        "Merton series with zero jump intensity matches Black-Scholes.",
        merton_degenerate_error < 1e-9,
        {"absolute_error": merton_degenerate_error},
        "absolute error < 1e-9",
    )

    # Merton Monte Carlo agrees with its series price inside the CI.
    merton_series_with_jumps = merton_series_price(
        100.0,
        100.0,
        0.5,
        0.2,
        0.04,
        OptionType.CALL,
        jump_intensity=0.5,
        jump_mean=-0.05,
        jump_vol=0.15,
        dividend_yield=0.0,
    )
    merton_mc = merton_mc_price(
        100.0,
        100.0,
        0.5,
        0.2,
        0.04,
        OptionType.CALL,
        jump_intensity=0.5,
        jump_mean=-0.05,
        jump_vol=0.15,
        n_paths=60_000,
        seed=4,
        dividend_yield=0.0,
    )
    merton_mc_error = abs(merton_mc["price"] - merton_series_with_jumps)
    record(
        "merton_mc_matches_series",
        "Merton Monte Carlo agrees with the series price.",
        merton_mc_error < 3 * merton_mc["standard_error"],
        {
            "absolute_error": merton_mc_error,
            "standard_error": merton_mc["standard_error"],
        },
        "error < 3 standard errors (60k paths, seed 4)",
    )

    # SVI calibration quality on the committed real option chain.
    svi_by_expiry = _svi_rmse_by_expiry()
    svi_ok = bool(svi_by_expiry)
    for expiry, result in svi_by_expiry.items():
        rmse = result["rmse_total_variance"]
        svi_ok = svi_ok and result["n"] >= 5 and rmse is not None and rmse < 5e-4
    record(
        "svi_real_chain_calibration_quality",
        "Per-expiry SVI calibration RMSE on the committed SPY chain.",
        svi_ok,
        svi_by_expiry,
        "all expiries calibrate with total-variance RMSE < 5e-4",
    )

    # Kupiec / Christoffersen VaR backtest on real SPY returns.
    var_result = _var_backtest_on_spy()
    record(
        "var_backtest_on_spy",
        "Kupiec and Christoffersen tests on a 95% rolling historical VaR.",
        (
            var_result["backtest_days"] == 1194
            and var_result["exceptions"] == 87
            and var_result["kupiec_p"] < 0.01
            and var_result["christoffersen_p"] > 0.05
        ),
        var_result,
        (
            "1194 days / 87 exceptions; Kupiec p < 0.01 "
            "(coverage rejected), Christoffersen p > 0.05 (no clustering)"
        ),
    )

    return {
        "report_name": "model_validation",
        "run_date": date.today().isoformat(),
        "all_passed": all(check["status"] == "pass" for check in checks),
        "checks": checks,
    }
