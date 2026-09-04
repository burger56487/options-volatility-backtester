from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.domain.enums import OptionType
from src.pricing.black_scholes import option_price
from src.pricing.requests import PricingRequest
from src.volatility_surface.stability import parameter_stability
from src.volatility_surface.surface import (
    VolSurface,
    SurfacePoint,
    surface_price,
)


def _surface() -> VolSurface:
    as_of = date(2026, 9, 4)
    return VolSurface(
        as_of=as_of,
        source="test",
        points=[
            SurfacePoint(
                expiry=as_of + timedelta(days=30),
                time_to_expiry=30 / 365,
                parameters={
                    "a": 0.03,
                    "b": 0.3,
                    "rho": -0.6,
                    "m": 0.0,
                    "sigma": 0.15,
                },
            ),
            SurfacePoint(
                expiry=as_of + timedelta(days=90),
                time_to_expiry=90 / 365,
                parameters={
                    "a": 0.05,
                    "b": 0.25,
                    "rho": -0.5,
                    "m": 0.0,
                    "sigma": 0.18,
                },
            ),
        ],
    )


def test_surface_hash_and_save(tmp_path):
    surface = _surface()
    assert surface.payload_hash
    path = surface.save(tmp_path / "surface.json")
    assert path.exists()


def test_interpolate_iv_rejects_non_positive_time():
    surface = _surface()
    with pytest.raises(ValueError, match="positive and finite"):
        surface.interpolate_iv(0.0, 0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        surface.interpolate_iv(0.0, float("nan"))


def test_surface_price_close_to_bsm_at_same_iv():
    surface = _surface()
    request = PricingRequest(
        spot=100.0,
        strike=100.0,
        time_to_expiry=30 / 365,
        risk_free_rate=0.04,
        dividend_yield=0.0,
        option_type=OptionType.CALL,
    )
    surface_result = surface_price(request, surface)
    forward = 100.0 * np.exp(0.04 * 30 / 365)
    iv = surface.interpolate_iv(np.log(100.0 / forward), 30 / 365)
    reference = option_price(
        spot=100.0,
        strike=100.0,
        time_to_expiry=30 / 365,
        risk_free_rate=0.04,
        volatility=iv,
        option_type="call",
        dividend_yield=0.0,
    )
    assert abs(surface_result.price - reference) < 1e-9


def test_parameter_stability_report():
    calibrations = pd.DataFrame(
        {
            "date": ["2026-09-01", "2026-09-02", "2026-09-03"],
            "rho": [-0.6, -0.55, -0.5],
            "sigma": [0.15, 0.16, 0.15],
        }
    )
    report = parameter_stability(
        calibrations,
        parameter_columns=["rho", "sigma"],
    )
    assert set(report["parameter"]) == {"rho", "sigma"}
    assert report.set_index("parameter").loc["rho", "range"] == pytest.approx(
        0.1
    )
