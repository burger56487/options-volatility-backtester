"""Portfolio Greeks with an explicit per-contract convention."""

from __future__ import annotations

from dataclasses import dataclass

from src.portfolio.account import Account
from src.portfolio.valuation import MarketSnapshot


@dataclass(frozen=True)
class PortfolioGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _total(account: Account, market: MarketSnapshot, field: str) -> float:
    values = {
        "delta": market.deltas,
        "gamma": market.gammas,
        "vega": market.vegas,
        "theta": market.thetas,
        "rho": market.rhos,
    }
    lookup = values[field]
    return sum(
        position.quantity * lookup[key]
        for key, position in account.positions.items()
        if key in lookup
    )


def calculate_portfolio_greeks(
    account: Account,
    market: MarketSnapshot,
) -> PortfolioGreeks:
    return PortfolioGreeks(
        delta=_total(account, market, "delta"),
        gamma=_total(account, market, "gamma"),
        vega=_total(account, market, "vega"),
        theta=_total(account, market, "theta"),
        rho=_total(account, market, "rho"),
    )
