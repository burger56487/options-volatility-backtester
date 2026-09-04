"""FastAPI application factory."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from src.pricing.black_scholes import price_and_greeks
from src.storage.repository import RunRepository

from .jobs import JobManager
from .schemas import (
    RunCreateRequest,
    RunCreateResponse,
    RunSummary,
    SurfacePriceRequest,
    SurfacePriceResponse,
    VanillaPriceRequest,
    VanillaPriceResponse,
)
from src.volatility_surface.surface import SurfacePoint, VolSurface
from src.pricing.cpp_backend import cpp_surface_prices, is_available


def create_app(
    repository: RunRepository,
    job_runner: Callable[[], dict] | None = None,
) -> FastAPI:
    """Build the API with a repository and an optional background runner."""
    manager = JobManager(repository)
    app = FastAPI(title="FICC Analytics Platform")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/pricing/vanilla", response_model=VanillaPriceResponse)
    def price_vanilla(request: VanillaPriceRequest) -> VanillaPriceResponse:
        result = price_and_greeks(
            spot=request.spot,
            strike=request.strike,
            time_to_expiry=request.time_to_expiry,
            risk_free_rate=request.risk_free_rate,
            volatility=request.volatility,
            option_type=request.option_type,
            dividend_yield=request.dividend_yield,
        )
        return VanillaPriceResponse(
            price=result.price,
            delta=result.delta,
            gamma=result.gamma,
            vega=result.vega,
            theta=result.theta,
            rho=result.rho,
        )

    @app.post("/pricing/surface", response_model=SurfacePriceResponse)
    def price_surface(request: SurfacePriceRequest) -> SurfacePriceResponse:
        as_of = request.as_of if request.as_of is not None else date.today()
        points = [
            SurfacePoint(
                expiry=as_of + timedelta(days=node.expiry_days),
                time_to_expiry=node.expiry_days / 365.0,
                parameters={
                    "a": node.a,
                    "b": node.b,
                    "rho": node.rho,
                    "m": node.m,
                    "sigma": node.sigma,
                },
            )
            for node in request.nodes
        ]
        surface = VolSurface(
            as_of=as_of,
            source="api",
            points=points,
        )
        if is_available():
            out = cpp_surface_prices(
                surface,
                spot=request.spot,
                risk_free_rate=request.risk_free_rate,
                dividend_yield=request.dividend_yield,
            )
            return SurfacePriceResponse(
                n_nodes=len(out["price"]),
                prices=[float(value) for value in out["price"]],
            )
        raise HTTPException(
            status_code=501,
            detail="C++ backend unavailable for surface pricing",
        )

    @app.post("/runs", response_model=RunCreateResponse)
    def create_run(request: RunCreateRequest) -> RunCreateResponse:
        if job_runner is None:
            raise HTTPException(
                status_code=400,
                detail="Background runner is not configured.",
            )
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        manager.submit(run_id, lambda: job_runner(request.payload))
        return RunCreateResponse(run_id=run_id, status="running")

    @app.get("/runs", response_model=list[RunSummary])
    def list_runs() -> list[RunSummary]:
        return [
            RunSummary(**summary) for summary in repository.list_runs()
        ]

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        record = repository.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return record

    return app
