"""Model calibration helpers for volatility surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.pricing.black_scholes import option_price
from src.volatility_surface.surface import VolSurface


@dataclass(frozen=True)
class ModelCalibrationResult:
    model: str
    parameters: dict[str, float]
    cost: float
    n_observations: int
    converged: bool
    message: str = ""


def calibrate_black_scholes_surface(
    surface: VolSurface,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    volatility_candidates: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
) -> ModelCalibrationResult:
    """Fit one constant BS volatility to all SVI surface nodes."""
    points = []
    for point in surface.points:
        for k in np.linspace(-0.1, 0.1, 9):
            iv = surface.interpolate_iv(float(k), point.time_to_expiry)
            market_price = option_price(
                spot=spot,
                strike=spot * np.exp(k),
                time_to_expiry=point.time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=iv,
                option_type="call",
                dividend_yield=dividend_yield,
            )
            points.append(
                {
                    "k": float(k),
                    "T": point.time_to_expiry,
                    "market_price": market_price,
                }
            )
    frame = pd.DataFrame(points)
    best = None
    for volatility in volatility_candidates:
        def residual(params):
            sigma = float(params[0])
            model = np.array(
                [
                    option_price(
                        spot=spot,
                        strike=spot * np.exp(row.k),
                        time_to_expiry=row.T,
                        risk_free_rate=risk_free_rate,
                        volatility=sigma,
                        option_type="call",
                        dividend_yield=dividend_yield,
                    )
                    for row in frame.itertuples()
                ]
            )
            return model - frame["market_price"].to_numpy()

        result = least_squares(
            residual,
            np.array([volatility]),
            bounds=([1e-4], [3.0]),
            max_nfev=2_000,
        )
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        raise RuntimeError("BS calibration failed.")
    return ModelCalibrationResult(
        model="black_scholes",
        parameters={"volatility": float(best.x[0])},
        cost=float(best.cost),
        n_observations=int(len(frame)),
        converged=bool(best.success),
    )


def model_comparison_report(
    surface: VolSurface,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    """Compare fitted BS against the SVI reference surface."""
    bs = calibrate_black_scholes_surface(
        surface,
        spot,
        risk_free_rate,
        dividend_yield,
    )
    frame = pd.DataFrame(
        [
            {
                "model": "black_scholes",
                "cost": bs.cost,
                "n_observations": bs.n_observations,
                "parameters": str(bs.parameters),
                "converged": bs.converged,
            },
            {
                "model": "svi_reference",
                "cost": 0.0,
                "n_observations": bs.n_observations,
                "parameters": "per-expiry SVI",
                "converged": True,
            },
        ]
    )
    return frame
