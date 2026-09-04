"""Pricing engine interface."""

from __future__ import annotations

from typing import Protocol

from .requests import PricingRequest
from .results import PricingResult


class PricingEngine(Protocol):
    """A pricing engine must price a request and return a result."""

    def price(self, request: PricingRequest) -> PricingResult:
        ...
