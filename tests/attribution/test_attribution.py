from src.attribution.core import (
    attribute_daily_leg,
    decompose_iv_change,
    waterfall_attribution,
)
from src.attribution.structures import (
    AttributionRow,
    DailyAttribution,
    classify_residual,
    save_daily_attribution,
)


def test_daily_leg_attribution_numbers():
    result = attribute_daily_leg(
        prev={
            "delta": 50.0,
            "gamma": 10.0,
            "vega": 100.0,
            "theta": -300.0,
            "rho": 0.0,
        },
        spot_change=2.0,
        iv_change=0.01,
        dt=1 / 365,
    )
    assert result["delta"] == 100.0
    assert result["gamma"] == 20.0
    assert result["vega"] == 1.0
    assert abs(result["theta"] + 300 / 365) < 1e-9


def test_waterfall_spot_only_change_matches_actual():
    result = waterfall_attribution(
        prev_spot=100.0,
        new_spot=105.0,
        prev_iv=0.2,
        new_iv=0.2,
        prev_t=0.5,
        new_t=0.5,
        prev_r=0.04,
        new_r=0.04,
        strike=100.0,
        option_type="call",
    )
    assert abs(
        result["actual_change"] - result["contributions"]["spot"]
    ) < 1e-8


def test_decompose_iv_change_separates_shape():
    import numpy as np

    k = np.linspace(-0.1, 0.1, 11)
    before = 0.20 + 0.10 * k
    after = 0.25 + 0.05 * k + 0.02 * k**2
    decomposition = decompose_iv_change(k, before, after)
    assert abs(decomposition["level_change"] - 0.05) < 1e-9
    assert abs(decomposition["skew_change"] + 0.05) < 1e-9
    assert abs(decomposition["curvature_change"] - 0.02) < 1e-9


def test_classify_and_save(tmp_path):
    attribution = DailyAttribution(
        rows=[
            AttributionRow(
                valuation_date="2026-09-04",
                instrument="SPY",
                factor="delta",
                amount=10.0,
                category="greek",
            )
        ]
    )
    save_daily_attribution(attribution, tmp_path)
    assert (tmp_path / "attribution_daily.csv").exists()
    assert (tmp_path / "attribution_summary.csv").exists()
    assert classify_residual(0.0, spot_move=0.01) == "clean"
    assert (
        classify_residual(5.0, spot_move=0.3) == "large_spot_jump"
    )
