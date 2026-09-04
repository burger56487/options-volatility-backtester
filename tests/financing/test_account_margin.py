from datetime import date, datetime

from src.financing.margin import estimate_account_margin
from src.portfolio.account import Account
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


def _stock_id():
    return InstrumentId(
        instrument_type=InstrumentType.STOCK,
        symbol="SPY",
    )


def test_long_option_and_short_stock_margin():
    account = Account(
        account_id="a1",
        run_id="r1",
        base_currency="USD",
        initial_capital=100_000.0,
    )
    timestamp = datetime(2026, 1, 2, 16, 0)
    account.apply_fill(
        Fill(
            fill_id="f1",
            order_id="o1",
            run_id="r1",
            timestamp=timestamp,
            instrument_id=_option_id(),
            side=Side.BUY,
            quantity=1,
            price=5.0,
            multiplier=100.0,
        )
    )
    account.apply_fill(
        Fill(
            fill_id="f2",
            order_id="o2",
            run_id="r1",
            timestamp=timestamp,
            instrument_id=_stock_id(),
            side=Side.SELL,
            quantity=100,
            price=50.0,
            multiplier=1.0,
        )
    )
    option_key = _option_id().key()
    stock_key = _stock_id().key()
    estimate = estimate_account_margin(
        account,
        market_prices={option_key: 5.0, stock_key: 50.0},
        underlying_spots={"SPY": 50.0},
    )
    assert estimate["initial_margin_total"] == 500.0 + 5000.0
    assert estimate["maintenance_margin_total"] == 5500.0
