"""Golden regression tests: recompute key model outputs and compare them
against values recorded on the 2026-09-04 evidence run.

Unlike smoke tests, every assertion here re-executes the underlying code on
committed data files, so a silent change in pricing, risk or calibration
logic fails the suite instead of passing by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.account_delta_hedge import (
    AccountHedgeConfig,
    run_rolling_account_straddle_backtest,
)
from src.market_data.underlying_data import load_price_data
from src.risk.backtest import (
    christoffersen_independence,
    kupiec_unconditional_coverage,
)
from src.volatility_surface.calibration import calibrate_svi
from src.volatility_surface.svi import svi_total_variance

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.golden


def test_golden_account_engine_rolling_pnl() -> None:
    """Account-engine rolling straddle reproduces the recorded total PnL."""
    price_data = load_price_data(
        PROJECT_ROOT / "data/raw/spy_daily_adjusted.csv"
    )
    result = run_rolling_account_straddle_backtest(
        price_data=price_data,
        config=AccountHedgeConfig(
            expiry_days=30,
            volatility=0.20,
        ),
        entry_spacing=60,
        minimum_history=252,
    )
    summary = result["summary"]
    assert summary["number_of_trades"] == 17
    assert summary["total_pnl"] == pytest.approx(-8038.7657, abs=0.05)
    assert summary["reconciliation_failures"] == 0


def test_golden_svi_rmse_scale() -> None:
    """Per-expiry SVI total-variance RMSE stays at the recorded scale."""
    path = PROJECT_ROOT / "outputs/real_option_chain/spy_option_chain_active.csv"
    frame = pd.read_csv(path, parse_dates=["expiry"])
    recorded = {
        "2026-09-08": (137, 6.447029431251334e-05),
        "2026-09-09": (138, 0.00016802203231817628),
        "2026-09-10": (139, 3.8906046023425044e-05),
        "2026-09-11": (240, 0.00016378207935656086),
        "2026-09-14": (45, 7.295457031127044e-05),
        "2026-09-15": (35, 5.111172676264133e-05),
    }
    computed: dict[str, tuple[int, float]] = {}
    for expiry, group in frame.groupby(frame["expiry"].dt.date):
        rows = group.dropna(subset=["iv"])
        rows = rows[rows["iv"] > 0]
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
        computed[str(expiry)] = (
            int(len(k)),
            float(np.sqrt(np.mean((model - w) ** 2))),
        )
    assert set(computed) == set(recorded)
    for expiry, (n, rmse) in recorded.items():
        computed_n, computed_rmse = computed[expiry]
        assert computed_n == n
        assert computed_rmse == pytest.approx(rmse, rel=1e-4, abs=1e-9)


def test_golden_var_backtest_p_values() -> None:
    """Kupiec/Christoffersen results reproduce the recorded evidence run."""
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
    assert kupiec["n"] == 1194
    assert kupiec["exceptions"] == 87
    assert kupiec["expected"] == pytest.approx(59.7)
    assert kupiec["p_value"] == pytest.approx(0.0006642785858820277, rel=1e-5)
    assert christoffersen["p_value"] == pytest.approx(
        0.781858926242632,
        rel=1e-5,
    )
