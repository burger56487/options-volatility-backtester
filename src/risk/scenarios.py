"""Factor shock scenarios and stress losses."""

from __future__ import annotations


def stress_loss(
    exposures: dict[str, float],
    shocks: dict[str, float],
) -> dict[str, float]:
    """Apply absolute factor shocks to exposures (PnL = -exposure*shock)."""
    by_factor = {
        factor: -exposure * shocks.get(factor, 0.0)
        for factor, exposure in exposures.items()
    }
    by_factor["total"] = sum(by_factor.values())
    return by_factor


def preset_scenario_shocks(kind: str) -> dict[str, float]:
    """Return factor shocks for named scenarios."""
    scenarios = {
        "equity_down_10": {"equity": -0.10, "volatility": 0.10},
        "vol_surge": {"equity": 0.0, "volatility": 0.20},
        "crash": {"equity": -0.20, "volatility": 0.30},
        "vol_surface_stress": {
            "equity": 0.0,
            "volatility": 0.20,
            "skew": -0.10,
            "curvature": 0.05,
            "term": 0.05,
        },
        "liquidity_widening": {
            "volatility": 0.05,
            "liquidity": 0.25,
        },
        "rate_up_100": {"rate": 0.01},
        "correlation_up": {"correlation": 0.20},
        "skew_flattening": {
            "skew": 0.15,
            "curvature": -0.05,
        },
    }
    if kind not in scenarios:
        raise KeyError(f"Unknown scenario: {kind}")
    return scenarios[kind]
