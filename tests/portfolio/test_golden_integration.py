"""Deterministic golden accounting scenario for the account engine."""

from datetime import date, datetime

from src.portfolio.account import Account
from src.portfolio.cash_ledger import calculate_cash_balance
from src.portfolio.fills import Fill
from src.portfolio.identifiers import (
    InstrumentId,
    InstrumentType,
    OptionType,
)
from src.portfolio.orders import Side
from src.portfolio.reconciliation import reconcile_pnl_bridge


def _option_id():
    return InstrumentId(
        instrument_type=InstrumentType.OPTION,
        symbol="SPY",
        expiry=date(2026, 3, 20),
        strike=500.0,
        option_type=OptionType.CALL,
    )


def _stock_id():
    return InstrumentId(
        instrument_type=InstrumentType.STOCK,
        symbol="SPY",
    )


def _fill(fill_id, instrument_id, side, quantity, price, multiplier, commission):
    return Fill(
        fill_id=fill_id,
        order_id=fill_id,
        run_id="golden",
        timestamp=datetime(2026, 1, 2, 16, 0),
        instrument_id=instrument_id,
        side=side,
        quantity=quantity,
        price=price,
        multiplier=multiplier,
        commission=commission,
    )


def test_golden_cash_and_position_path():
    account = Account(
        account_id="golden",
        run_id="golden",
        base_currency="USD",
        initial_capital=100_000.0,
    )
    option_key = _option_id().key()
    stock_key = _stock_id().key()

    # Buy 1 call at 5.00 (x100) with 1.00 commission.
    account.apply_fill(
        _fill("f1", _option_id(), Side.BUY, 1, 5.0, 100.0, 1.0)
    )
    assert account.cash == 100_000 - 500 - 1
    assert account.positions[option_key].quantity == 1

    # Hedge: buy 50 shares at 100.00, per-share commission 0.005 (0.25 < min 1).
    account.apply_fill(
        _fill("f2", _stock_id(), Side.BUY, 50, 100.0, 1.0, 1.0)
    )
    assert account.cash == 100_000 - 501 - 5001
    assert account.positions[stock_key].quantity == 50

    # Mark at call=5, stock=100: equity = cash + 500 + 5000 = 100_000 - 2.
    prices = {option_key: 5.0, stock_key: 100.0}
    assert account.equity(prices) == 100_000 - 2
    assert calculate_cash_balance(account.cash_ledger) == account.cash

    # Close the stock at 105: realized +250, cash gains 5250 - 1.
    account.apply_fill(
        _fill("f3", _stock_id(), Side.SELL, 50, 105.0, 1.0, 1.0)
    )
    assert account.positions[stock_key].quantity == 0
    assert account.positions[stock_key].realised_pnl == 250.0
    assert account.cash == 100_000 - 501 - 5001 + 5250 - 1

    result = reconcile_pnl_bridge(account, prices)
    assert result.passed
    assert result.difference == 0.0
