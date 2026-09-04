"""End-to-end C++ backend demo: surface repricing, scenario PnL, VaR."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from src.pricing.cpp_backend import (
    cpp_portfolio_var,
    cpp_scenario_pnl,
    cpp_surface_prices,
    is_available,
)
from src.volatility_surface.surface import SurfacePoint, VolSurface


def main() -> None:
    if not is_available():
        raise SystemExit("C++ backend not available; rebuild bs_kernels.dll.")
    as_of = date(2026, 9, 4)
    surface = VolSurface(
        as_of=as_of,
        source="demo",
        points=[
            SurfacePoint(
                expiry=as_of + timedelta(days=d),
                time_to_expiry=d / 365,
                parameters={
                    "a": 0.03,
                    "b": 0.12,
                    "rho": -0.3,
                    "m": 0.0,
                    "sigma": 0.12,
                },
            )
            for d in (30, 60, 90)
        ],
    )
    prices = cpp_surface_prices(
        surface,
        spot=100.0,
        risk_free_rate=0.04,
        moneyness_grid=np.linspace(-0.1, 0.1, 9),
    )
    rows = prices["surface_points"]
    pnl = cpp_scenario_pnl(
        np.array([r["spot"] for r in rows]),
        np.array([r["strike"] for r in rows]),
        np.array([r["t"] for r in rows]),
        0.04,
        0.0,
        np.array([r["vol"] for r in rows]),
        np.array(["call"] * len(rows)),
        spot_shock=-0.1,
        vol_shock=0.05,
    )
    exposures = np.array([100.0, -50.0])
    covariance = np.array([[0.01, 0.0], [0.0, 0.04]])
    var = cpp_portfolio_var(exposures, covariance, z_score=1.645)
    evidence = {
        "surface_nodes": int(len(prices["price"])),
        "surface_total_pnl_shock": float(np.sum(pnl)),
        "portfolio_var": var["var"],
        "euler_sum": float(np.sum(var["contributions"])),
    }
    output = Path("outputs") / "cpp_backend_demo.json"
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
