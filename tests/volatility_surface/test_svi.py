import numpy as np
import pandas as pd

from src.pricing.black_scholes import option_price
from src.volatility_surface.arbitrage import (
    check_butterfly_calls,
    check_calendar_screen,
)
from src.volatility_surface.calibration import calibrate_svi
from src.volatility_surface.svi import svi_total_variance


def _synthetic(k):
    return svi_total_variance(
        k, a=0.04, b=0.3, rho=-0.6, m=0.02, sigma=0.15
    )


def test_svi_calibration_recovers_parameters():
    k = np.linspace(-0.2, 0.2, 15)
    w = _synthetic(k) + np.random.default_rng(1).normal(0, 1e-4, len(k))
    result = calibrate_svi(k, w, minimum_points=5)
    assert result.n_points == 15
    assert abs(result.parameters["rho"] - (-0.6)) < 0.05
    assert abs(result.parameters["sigma"] - 0.15) < 0.05


def test_butterfly_holds_on_bsm_prices():
    dates = pd.date_range("2026-01-02", periods=60, freq="B")
    expiry = dates[-1]
    rows = []
    spot = 100.0
    for strike in np.linspace(85, 115, 16):
        mid = option_price(
            spot=spot,
            strike=strike,
            time_to_expiry=60 / 365,
            risk_free_rate=0.04,
            volatility=0.25,
            option_type="call",
            dividend_yield=0.0,
        )
        rows.append(
            {
                "expiry": expiry,
                "strike": float(strike),
                "option_type": "call",
                "mid": mid,
            }
        )
    quotes = pd.DataFrame(rows)
    violations = check_butterfly_calls(quotes, risk_free_rate=0.04)
    assert violations.empty


def test_calendar_screen_uses_total_variance():
    surface = pd.DataFrame(
        [
            {"expiry": 1, "log_moneyness": 0.0, "total_variance": 0.10},
            {"expiry": 1, "log_moneyness": 0.1, "total_variance": 0.12},
            {"expiry": 2, "log_moneyness": 0.0, "total_variance": 0.08},
            {"expiry": 2, "log_moneyness": 0.1, "total_variance": 0.13},
        ]
    )
    violations = check_calendar_screen(surface)
    assert len(violations) == 1
