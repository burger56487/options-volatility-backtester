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


def _option_id():
    return InstrumentId(
        instrument_type=InstrumentType.OPTION,
        symbol="SPY",
        expiry=date(2026, 3, 20),
        strike=500.0,
        option_type=OptionType.CALL,
    )


def _account():
    return Account(
        account_id="a1",
        run_id="r1",
        base_currency="USD",
        initial_capital=100_000.0,
    )


def _fill(side: Side, quantity: float, price: float, commission: float = 1.0):
    return Fill(
        fill_id="f1",
        order_id="o1",
        run_id="r1",
        timestamp=datetime(2026, 1, 2, 16, 0),
        instrument_id=_option_id(),
        side=side,
        quantity=quantity,
        price=price,
        multiplier=100.0,
        commission=commission,
    )


def test_buy_option_reduces_cash_and_creates_position():
    account = _account()
    account.apply_fill(_fill(Side.BUY, 1, 5.0))
    assert account.cash == 100_000 - 500 - 1
    assert account.positions[_option_id().key()].quantity == 1
    assert calculate_cash_balance(account.cash_ledger) == account.cash


def test_cash_ledger_matches_account_cash_after_roundtrip():
    account = _account()
    account.apply_fill(_fill(Side.BUY, 1, 5.0))
    account.apply_fill(_fill(Side.SELL, 1, 7.0))
    assert account.positions[_option_id().key()].quantity == 0
    assert account.cash == 100_000 - 500 - 1 + 700 - 1
    assert calculate_cash_balance(account.cash_ledger) == account.cash
