"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VanillaPriceRequest(BaseModel):
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    time_to_expiry: float = Field(ge=0)
    risk_free_rate: float = 0.04
    volatility: float = Field(gt=0)
    option_type: str = Field(pattern="^(call|put)$")
    dividend_yield: float = 0.0


class VanillaPriceResponse(BaseModel):
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class SurfaceNodeParams(BaseModel):
    expiry_days: int = Field(gt=0)
    a: float = Field(default=0.03)
    b: float = Field(default=0.1)
    rho: float = Field(default=0.0, ge=-1, le=1)
    m: float = Field(default=0.0)
    sigma: float = Field(default=0.12, gt=0)


class SurfacePriceRequest(BaseModel):
    spot: float = Field(gt=0)
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.0
    nodes: list[SurfaceNodeParams] = Field(min_length=1)


class SurfacePriceResponse(BaseModel):
    n_nodes: int
    prices: list[float]


class RunCreateRequest(BaseModel):
    name: str = "experiment"
    payload: dict[str, Any] = Field(default_factory=dict)


class RunCreateResponse(BaseModel):
    run_id: str
    status: str


class RunSummary(BaseModel):
    run_id: str
    created_at: str
    status: str
