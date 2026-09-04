"""Deterministic execution engine with partial fills and stale checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import uuid4

from src.portfolio.account import Account
from src.portfolio.orders import Order, OrderStatus, Side
from .commission import CommissionSchedule, commission_for
from .fill_model import (
    ExecutionParameters,
    make_fill,
)
from .market_snapshot import MarketSnapshot


@dataclass
class ExecutionResult:
    fills: list = field(default_factory=list)
    status: OrderStatus = OrderStatus.CREATED
    unfilled_quantity: float = 0.0
    rejection_reason: str | None = None


class ExecutionEngine:
    """Executes one order against a snapshot and applies fills to account."""

    def __init__(
        self,
        commission_schedule: CommissionSchedule = CommissionSchedule(),
        parameters: ExecutionParameters = ExecutionParameters(),
        run_id: str = "run",
    ) -> None:
        self.commission_schedule = commission_schedule
        self.parameters = parameters
        self.run_id = run_id
        self.fill_counter = 0

    def _next_fill_id(self) -> str:
        self.fill_counter += 1
        return f"fill-{self.run_id}-{self.fill_counter}"

    def execute(
        self,
        order: Order,
        snapshot: MarketSnapshot,
        account: Account | None = None,
        available_quantity: float | None = None,
        multiplier: float = 1.0,
        latency_seconds: float = 0.0,
    ) -> ExecutionResult:
        if snapshot.is_stale(self.parameters.max_quote_age_seconds):
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "stale_quote"
            return ExecutionResult(
                status=OrderStatus.REJECTED,
                rejection_reason="stale_quote",
            )

        max_fill = (
            order.quantity
            if available_quantity is None
            else min(order.quantity, available_quantity)
        )
        if max_fill <= 0:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "no_available_liquidity"
            return ExecutionResult(
                status=OrderStatus.REJECTED,
                rejection_reason="no_available_liquidity",
            )

        if (
            order.order_type.value == "limit"
            and order.limit_price is not None
        ):
            if (
                order.side == Side.BUY
                and snapshot.ask > order.limit_price
            ) or (
                order.side == Side.SELL
                and snapshot.bid < order.limit_price
            ):
                order.status = OrderStatus.CANCELLED
                return ExecutionResult(
                    status=OrderStatus.CANCELLED,
                    unfilled_quantity=order.quantity,
                    rejection_reason="limit_not_touchable",
                )

        partial = max_fill < order.quantity
        if partial and not self.parameters.allow_partial_fill:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "partial_fill_not_allowed"
            return ExecutionResult(
                status=OrderStatus.REJECTED,
                rejection_reason="partial_fill_not_allowed",
            )

        commission = commission_for(
            quantity=max_fill,
            multiplier=multiplier,
            schedule=self.commission_schedule,
        )
        fill_timestamp = (
            snapshot.timestamp
            + timedelta(seconds=latency_seconds)
        )
        fill = make_fill(
            order=order,
            run_id=self.run_id,
            fill_id=self._next_fill_id(),
            timestamp=fill_timestamp,
            quantity=max_fill,
            multiplier=multiplier,
            snapshot=snapshot,
            parameters=self.parameters,
            commission=commission,
        )
        if account is not None:
            account.apply_fill(fill)

        order.status = (
            OrderStatus.PARTIALLY_FILLED
            if partial
            else OrderStatus.FILLED
        )
        return ExecutionResult(
            fills=[fill],
            status=order.status,
            unfilled_quantity=(
                order.quantity - max_fill if partial else 0.0
            ),
        )
