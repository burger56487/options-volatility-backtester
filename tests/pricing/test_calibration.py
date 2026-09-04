from datetime import date, timedelta

from src.pricing.calibration import (
    calibrate_black_scholes_surface,
    model_comparison_report,
)
from src.volatility_surface.surface import SurfacePoint, VolSurface


def _surface() -> VolSurface:
    as_of = date(2026, 9, 4)
    return VolSurface(
        as_of=as_of,
        source="test",
        points=[
            SurfacePoint(
                expiry=as_of + timedelta(days=d),
                time_to_expiry=d / 365,
                parameters={
                    "a": 0.03,
                    "b": 0.1,
                    "rho": -0.1,
                    "m": 0.0,
                    "sigma": 0.1,
                },
            )
            for d in (30, 60, 90)
        ],
    )


def test_bs_calibration_converges_on_surface():
    surface = _surface()
    result = calibrate_black_scholes_surface(
        surface,
        spot=100.0,
        risk_free_rate=0.04,
    )
    assert result.converged
    assert result.n_observations > 0
    assert 0.0 < result.parameters["volatility"] < 1.0


def test_model_comparison_report_lists_models():
    report = model_comparison_report(
        _surface(),
        spot=100.0,
        risk_free_rate=0.04,
    )
    assert set(report["model"]) == {
        "black_scholes",
        "svi_reference",
    }
