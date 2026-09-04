"""Volatility surface object with versioning and pricing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from src.pricing.black_scholes import price_and_greeks
from src.pricing.requests import PricingRequest
from src.pricing.results import PricingResult
from .svi import svi_total_variance


@dataclass(frozen=True)
class SurfacePoint:
    expiry: date
    time_to_expiry: float
    parameters: dict[str, float]


def _hash_payload(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class VolSurface:
    as_of: date
    points: list[SurfacePoint] = field(default_factory=list)
    source: str = "synthetic"
    version: str = "0.1.0"
    payload_hash: str = ""

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("VolSurface requires at least one point.")
        object.__setattr__(
            self,
            "payload_hash",
            _hash_payload(self.to_dict()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "source": self.source,
            "version": self.version,
            "points": [
                {
                    "expiry": point.expiry.isoformat(),
                    "time_to_expiry": point.time_to_expiry,
                    "parameters": point.parameters,
                }
                for point in self.points
            ],
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    **self.to_dict(),
                    "payload_hash": self.payload_hash,
                },
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        return output

    def interpolate_iv(
        self,
        log_moneyness: float,
        time_to_expiry: float,
    ) -> float:
        """Interpolate implied vol linearly in total variance across maturities."""
        points = sorted(self.points, key=lambda p: p.time_to_expiry)
        if time_to_expiry <= points[0].time_to_expiry:
            chosen = points[0]
        elif time_to_expiry >= points[-1].time_to_expiry:
            chosen = points[-1]
        else:
            for i in range(len(points) - 1):
                t1, t2 = (
                    points[i].time_to_expiry,
                    points[i + 1].time_to_expiry,
                )
                if t1 <= time_to_expiry <= t2:
                    w1 = svi_total_variance(
                        np.array([log_moneyness]),
                        **points[i].parameters,
                    )[0]
                    w2 = svi_total_variance(
                        np.array([log_moneyness]),
                        **points[i + 1].parameters,
                    )[0]
                    fraction = (time_to_expiry - t1) / (t2 - t1)
                    total_variance = w1 + fraction * (w2 - w1)
                    return float(
                        math.sqrt(max(total_variance, 0.0) / time_to_expiry)
                    )
        return float(
            math.sqrt(
                max(
                    svi_total_variance(
                        np.array([log_moneyness]),
                        **chosen.parameters,
                    )[0],
                    0.0,
                )
                / chosen.time_to_expiry
            )
        )


def surface_price(
    request: PricingRequest,
    surface: VolSurface,
) -> PricingResult:
    """Price an option from the interpolated surface implied volatility."""
    if request.volatility is not None:
        raise ValueError("surface_price ignores request volatility input.")
    forward = request.spot * math.exp(
        (request.risk_free_rate - request.dividend_yield)
        * request.time_to_expiry
    )
    log_moneyness = math.log(request.strike / forward)
    iv = surface.interpolate_iv(log_moneyness, request.time_to_expiry)
    result = price_and_greeks(
        spot=request.spot,
        strike=request.strike,
        time_to_expiry=request.time_to_expiry,
        risk_free_rate=request.risk_free_rate,
        volatility=iv,
        option_type=request.option_type.value,
        dividend_yield=request.dividend_yield,
    )
    return PricingResult(
        price=result.price,
        delta=result.delta,
        gamma=result.gamma,
        vega=result.vega,
        theta=result.theta,
        rho=result.rho,
        method="volatility_surface",
    )
