"""One-shot evidence run: numbers for CV claims (2026-09-04)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from dataclasses import replace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.domain.enums import OptionType
from src.pricing.black_scholes import option_price
from src.pricing.merton import merton_mc_price, merton_series_price
from src.pricing.heston import heston_mc_price
from src.pricing.binomial import crr_price
from src.pricing.finite_difference import finite_difference_price
from src.pricing.requests import PricingRequest
from src.risk.backtest import (
    christoffersen_independence,
    kupiec_unconditional_coverage,
)
from src.risk.measure import historical_var
from src.statistics import block_bootstrap_intervals
from src.volatility_surface.calibration import calibrate_svi
from src.volatility_surface.svi import svi_total_variance


def svi_rmse_by_expiry() -> dict:
    path = Path("outputs/real_option_chain/spy_option_chain_active.csv")
    if not path.exists():
        return {"error": "active chain csv missing; run real-chain script first"}
    frame = pd.read_csv(path, parse_dates=["expiry"])
    result = {}
    for expiry, group in frame.groupby(frame["expiry"].dt.date):
        rows = group.dropna(subset=["iv"])
        rows = rows[(rows["iv"] > 0)]
        k = rows["log_moneyness"].to_numpy()
        w = (rows["iv"] ** 2 * rows["time_to_expiry"]).to_numpy()
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
            "rmse_total_variance": float(
                np.sqrt(np.mean((model - w) ** 2))
            ),
            "rho": params["rho"],
        }
    return result


def pricing_convergence() -> dict:
    reference = option_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        volatility=0.25,
        option_type="call",
        dividend_yield=0.01,
    )
    request = PricingRequest(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        volatility=0.25,
        option_type=OptionType.CALL,
    )
    crr = {
        steps: abs(crr_price(replace(request, steps=steps)).price - reference)
        for steps in (100, 200, 400, 800)
    }
    fd = {}
    for steps in (200, 400):
        fd[steps] = abs(
            finite_difference_price(
                request, theta=0.5, x_steps=2 * steps, time_steps=steps
            ).price
            - reference
        )
    merton = {}
    series = merton_series_price(
        100.0,
        100.0,
        0.5,
        0.2,
        0.04,
        OptionType.CALL,
        0.5,
        -0.05,
        0.15,
    )
    for n_paths in (10_000, 50_000, 200_000):
        mc = merton_mc_price(
            100.0,
            100.0,
            0.5,
            0.2,
            0.04,
            OptionType.CALL,
            0.5,
            -0.05,
            0.15,
            n_paths=n_paths,
            seed=7,
        )
        merton[str(n_paths)] = {
            "price": mc["price"],
            "series": series,
            "error": abs(mc["price"] - series),
            "ci_width": mc["ci_high"] - mc["ci_low"],
        }
    heston_reference = option_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.04,
        volatility=0.2,
        option_type="call",
        dividend_yield=0.0,
    )
    heston = heston_mc_price(
        100.0,
        100.0,
        0.5,
        1.0,
        0.04,
        1e-6,
        0.0,
        0.04,
        0.04,
        OptionType.CALL,
        dividend_yield=0.0,
        n_paths=80_000,
        n_steps=30,
        seed=3,
    )
    return {
        "bs_reference": reference,
        "crr_absolute_error": crr,
        "crank_nicolson_absolute_error": fd,
        "merton_mc": merton,
        "heston_bs_reference": heston_reference,
        "heston_degenerate_error": abs(heston["price"] - heston_reference),
        "heston_ci_width": heston["ci_high"] - heston["ci_low"],
    }


def var_backtest_on_spy() -> dict:
    frame = pd.read_csv(
        "data/raw/spy_daily_adjusted.csv",
        parse_dates=["Date"],
    ).set_index("Date")
    returns = frame["Close"].pct_change().dropna()
    var = historical_var(returns.tail(500), confidence_level=0.95)
    rolling_var = (
        -returns.rolling(60).apply(
            lambda x: np.quantile(x, 0.05), raw=True
        )
        .shift(1)
    )
    window = returns.iloc[60:]
    thresholds = rolling_var.loc[window.index]
    valid = pd.concat([window, thresholds], axis=1).dropna()
    kupiec = kupiec_unconditional_coverage(
        valid.iloc[:, 0].to_numpy(),
        valid.iloc[:, 1].to_numpy(),
    )
    christoffersen = christoffersen_independence(
        valid.iloc[:, 0].to_numpy(),
        valid.iloc[:, 1].to_numpy(),
    )
    return {
        "full_sample_95_var_daily": var.var,
        "backtest_days": kupiec["n"],
        "exceptions": kupiec["exceptions"],
        "expected": kupiec["expected"],
        "kupiec_p": kupiec["p_value"],
        "christoffersen_p": christoffersen["p_value"],
    }


def bootstrap_ci() -> dict:
    data = load_spy()
    trade_returns = legacy_trade_returns(data)
    intervals = block_bootstrap_intervals(
        trade_returns,
        block_size=5,
        n_samples=2000,
        seed=1,
        confidence_level=0.95,
        trades_per_year=8.4,
    )
    daily_returns = data["Close"].pct_change().dropna().tail(500)
    daily_ci = block_bootstrap_intervals(
        daily_returns,
        block_size=8,
        n_samples=2000,
        seed=2,
        confidence_level=0.95,
        trades_per_year=252,
    )
    return {
        "legacy_trade_sharpe_ci": {
            "low": intervals["sharpe_like_ratio_ci_low"],
            "high": intervals["sharpe_like_ratio_ci_high"],
            "width": intervals["sharpe_like_ratio_ci_high"]
            - intervals["sharpe_like_ratio_ci_low"],
        },
        "spy_daily_annualized_sharpe_ci": {
            "low": daily_ci["annualized_sharpe_ci_low"],
            "high": daily_ci["annualized_sharpe_ci_high"],
            "width": daily_ci["annualized_sharpe_ci_high"]
            - daily_ci["annualized_sharpe_ci_low"],
        },
    }


def load_spy() -> pd.DataFrame:
    frame = pd.read_csv(
        "data/raw/spy_daily_adjusted.csv",
        parse_dates=["Date"],
    ).set_index("Date")
    return pd.DataFrame({"Close": frame["Close"]}, index=frame.index)


def legacy_trade_returns(data: pd.DataFrame) -> pd.Series:
    from src.backtest.long_straddle_backtest import LongStraddleBacktestConfig
    from src.backtest.rolling_backtest import (
        RollingBacktestConfig,
        run_rolling_long_straddle_backtest,
    )

    result = run_rolling_long_straddle_backtest(
        data,
        LongStraddleBacktestConfig(days_to_expiry=30, delta_threshold=5.0),
        RollingBacktestConfig(entry_spacing_trading_days=30),
    )
    return result.trade_results["trade_return"]


def main() -> None:
    evidence = {
        "run_date": "2026-09-04",
        "svi_rmse_by_expiry": svi_rmse_by_expiry(),
        "pricing_convergence": pricing_convergence(),
        "var_backtest_on_spy": var_backtest_on_spy(),
        "bootstrap_ci": bootstrap_ci(),
    }
    output = Path("outputs") / "evidence_20260904.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, default=str))
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
