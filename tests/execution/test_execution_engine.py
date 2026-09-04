from datetime import datetime

import pytest

from src.execution.commission import (
    CommissionSchedule,
    commission_for,
)
from src.execution.engine import ExecutionEngine
from src.execution.fill_model import ExecutionParameters
from src.execution.market_snapshot import MarketSnapshot
from src.execution.metrics import execution_metrics
from src.portfolio.account import Account
from src.portfolio.identifiers import (
    InstrumentId,
    InstrumentType,
)
from src.portfolio.orders import Order, OrderStatus, OrderType, Side


def _stock_id():
    return InstrumentId(
        instrument_type=InstrumentType.STOCK,
        symbol="SPY",
    )


def _snapshot(stale: bool = False):
    return MarketSnapshot(
        timestamp=datetime(2026, 1, 2, 14, 0),
        symbol="SPY",
        bid=99.0,
        ask=101.0,
        mid=100.0,
        orderbook_mode="orderbook",
        quote_age_seconds=500.0 if stale else 5.0,
    )


def test_option_commission_uses_per_contract():
    schedule = CommissionSchedule(
        per_share=0.005,
        per_contract=0.65,
        minimum=1.0,
    )
    assert commission_for(2, 100.0, schedule) == 1.3


def test_stale_quote_is_rejected():
    engine = ExecutionEngine()
    order = Order(
        order_id="o1",
        run_id="r1",
        timestamp=datetime(2026, 1, 2, 14, 0),
        instrument_id=_stock_id(),
        side=Side.BUY,
        quantity=100,
    )
    result = engine.execute(order, _snapshot(stale=True))
    assert result.status == OrderStatus.REJECTED
    assert result.rejection_reason == "stale_quote"


def test_partial_fill_when_liquidity_limited():
    engine = ExecutionEngine()
    order = Order(
        order_id="o1",
        run_id="r1",
        timestamp=datetime(2026, 1, 2, 14, 0),
        instrument_id=_stock_id(),
        side=Side.BUY,
        quantity=200,
    )
    result = engine.execute(
        order,
        _snapshot(),
        available_quantity=50,
    )
    assert result.status == OrderStatus.PARTIALLY_FILLED
    assert result.unfilled_quantity == 150
    assert len(result.fills) == 1
    assert result.fills[0].quantity == 50


def test_limit_order_cancelled_when_not_touchable():
    engine = ExecutionEngine()
    order = Order(
        order_id="o1",
        run_id="r1",
        timestamp=datetime(2026, 1, 2, 14, 0),
        instrument_id=_stock_id(),
        side=Side.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=99.5,
    )
    result = engine.execute(order, _snapshot())
    assert result.status == OrderStatus.CANCELLED


def test_marketable_buy_limit_fills_at_or_below_limit():
    engine = ExecutionEngine()
    order = Order(
        order_id="o1",
        run_id="r1",
        timestamp=datetime(2026, 1, 2, 14, 0),
        instrument_id=_stock_id(),
        side=Side.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=101.0,  # exactly the ask
    )
    result = engine.execute(order, _snapshot())
    assert result.status == OrderStatus.FILLED
    assert result.fills[0].price <= 101.0


def test_marketable_sell_limit_fills_at_or_above_limit():
    engine = ExecutionEngine()
    order = Order(
        order_id="o1",
        run_id="r1",
        timestamp=datetime(2026, 1, 2, 14, 0),
        instrument_id=_stock_id(),
        side=Side.SELL,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=99.0,  # exactly the bid
    )
    result = engine.execute(order, _snapshot())
    assert result.status == OrderStatus.FILLED
    assert result.fills[0].price >= 99.0


def test_market_impact_scales_with_multiplied_units():
    engine = ExecutionEngine(
        parameters=ExecutionParameters(
            impact_coefficient=1e-4,
            impact_exponent=0.5,
        )
    )
    order = Order(
        order_id="o1",
        run_id="r1",
        timestamp=datetime(2026, 1, 2, 14, 0),
        instrument_id=_stock_id(),
        side=Side.BUY,
        quantity=1,
    )
    result = engine.execute(
        order,
        _snapshot(),
        multiplier=100,
    )
    units = 100.0
    impact_per_share = 1e-4 * units**0.5
    assert result.fills[0].market_impact_cost == pytest.approx(
        impact_per_share * units
    )


def test_execution_updates_account_and_metrics():
    account = Account(
        account_id="a1",
        run_id="r1",
        base_currency="USD",
        initial_capital=100_000.0,
    )
    engine = ExecutionEngine()
    order = Order(
        order_id="o1",
        run_id="r1",
        timestamp=datetime(2026, 1, 2, 14, 0),
        instrument_id=_stock_id(),
        side=Side.BUY,
        quantity=100,
    )
    result = engine.execute(order, _snapshot(), account=account)
    key = _stock_id().key()
    assert account.positions[key].quantity == 100
    assert result.fills[0].price > 100.0  # buy pays half-spread + slippage
    metrics = execution_metrics(
        requested_quantity=order.quantity,
        fills=result.fills,
    )
    assert metrics.fill_rate == 1.0
    assert metrics.total_slippage_cost > 0.0
