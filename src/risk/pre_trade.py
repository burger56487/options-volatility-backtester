"""Pre-trade risk checks, including a binary-search max-fill quantity."""

from __future__ import annotations

from copy import deepcopy

from src.portfolio.account import Account
from src.portfolio.fills import Fill
from src.portfolio.valuation import MarketSnapshot
from src.risk.exposures import calculate_portfolio_greeks
from src.risk.limits import (
    LimitCheckResult,
    RiskLimits,
    check_portfolio_limits,
)


def simulate_post_trade_limit_check(
    account: Account,
    hypothetical_fill: Fill,
    market: MarketSnapshot,
    limits: RiskLimits,
    daily_pnl: float,
) -> LimitCheckResult:
    simulated_account = deepcopy(account)
    simulated_account.apply_fill(hypothetical_fill)
    greeks = calculate_portfolio_greeks(simulated_account, market)
    return check_portfolio_limits(
        gross_exposure=simulated_account.gross_exposure(market.prices),
        leverage=simulated_account.leverage(market.prices),
        delta=greeks.delta,
        gamma=greeks.gamma,
        vega=greeks.vega,
        daily_pnl=daily_pnl,
        drawdown=simulated_account.drawdown(market.prices),
        cash=simulated_account.cash,
        limits=limits,
    )


def max_allowed_fill_quantity(
    account: Account,
    order_quantity: float,
    make_fill: "callable",
    market: MarketSnapshot,
    limits: RiskLimits,
    daily_pnl: float,
) -> float:
    """Binary-search the largest quantity that passes pre-trade limits."""
    if order_quantity <= 0:
        return 0.0
    low, high = 0.0, order_quantity
    best = 0.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if mid == low or mid == high:
            break
        check = simulate_post_trade_limit_check(
            account=account,
            hypothetical_fill=make_fill(mid),
            market=market,
            limits=limits,
            daily_pnl=daily_pnl,
        )
        if check.allowed:
            best = mid
            low = mid
        else:
            high = mid
    return best
