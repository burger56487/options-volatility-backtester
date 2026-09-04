"""Portfolio risk limits and breach reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LimitAction(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"
    REDUCE = "reduce"
    LIQUIDATE = "liquidate"
    HALT = "halt"


@dataclass(frozen=True)
class RiskLimits:
    max_gross_exposure: float
    max_leverage: float
    max_abs_delta: float
    max_abs_gamma: float
    max_abs_vega: float
    max_daily_loss: float
    max_drawdown: float
    min_cash_buffer: float


@dataclass(frozen=True)
class LimitBreach:
    name: str
    current_value: float
    limit_value: float
    action: LimitAction
    message: str


@dataclass
class LimitCheckResult:
    allowed: bool
    breaches: list[LimitBreach] = field(default_factory=list)


BLOCKING_ACTIONS = {
    LimitAction.REJECT,
    LimitAction.LIQUIDATE,
    LimitAction.HALT,
}


def check_portfolio_limits(
    gross_exposure: float,
    leverage: float,
    delta: float,
    gamma: float,
    vega: float,
    daily_pnl: float,
    drawdown: float,
    cash: float,
    limits: RiskLimits,
) -> LimitCheckResult:
    breaches = []
    if gross_exposure > limits.max_gross_exposure:
        breaches.append(
            LimitBreach(
                "gross_exposure",
                gross_exposure,
                limits.max_gross_exposure,
                LimitAction.REJECT,
                "gross exposure exceeds limit",
            )
        )
    if leverage > limits.max_leverage:
        breaches.append(
            LimitBreach(
                "leverage",
                leverage,
                limits.max_leverage,
                LimitAction.REJECT,
                "leverage exceeds limit",
            )
        )
    for name, value, limit in [
        ("delta", abs(delta), limits.max_abs_delta),
        ("gamma", abs(gamma), limits.max_abs_gamma),
        ("vega", abs(vega), limits.max_abs_vega),
    ]:
        if value > limit:
            breaches.append(
                LimitBreach(
                    name,
                    value,
                    limit,
                    LimitAction.REDUCE,
                    f"{name} exceeds limit",
                )
            )
    if daily_pnl < -limits.max_daily_loss:
        breaches.append(
            LimitBreach(
                "daily_loss",
                daily_pnl,
                -limits.max_daily_loss,
                LimitAction.HALT,
                "daily loss exceeds limit",
            )
        )
    if drawdown < -limits.max_drawdown:
        breaches.append(
            LimitBreach(
                "drawdown",
                drawdown,
                -limits.max_drawdown,
                LimitAction.LIQUIDATE,
                "drawdown exceeds limit",
            )
        )
    if cash < limits.min_cash_buffer:
        breaches.append(
            LimitBreach(
                "cash_buffer",
                cash,
                limits.min_cash_buffer,
                LimitAction.REJECT,
                "cash buffer insufficient",
            )
        )
    return LimitCheckResult(
        allowed=not any(
            breach.action in BLOCKING_ACTIONS
            for breach in breaches
        ),
        breaches=breaches,
    )
