from datetime import date, datetime

from src.portfolio.account import Account
from src.portfolio.fills import Fill
from src.portfolio.identifiers import (
    InstrumentId,
    InstrumentType,
    OptionType,
)
from src.portfolio.orders import Side


def test_call_option_settlement():
    account = Account(
        account_id="a1",
        run_id="r1",
        base_currency="USD",
        initial_capital=10_000.0,
    )
    option_id = InstrumentId(
        instrument_type=InstrumentType.OPTION,
        symbol="SPY",
        expiry=date(2026, 3, 20),
        strike=500.0,
        option_type=OptionType.CALL,
    )
    account.apply_fill(
        Fill(
            fill_id="f1",
            order_id="o1",
            run_id="r1",
            timestamp=datetime(2026, 1, 2),
            instrument_id=option_id,
            side=Side.BUY,
            quantity=1,
            price=5.0,
            multiplier=100.0,
            commission=1.0,
        )
    )
    account.settle_expired_options(
        timestamp=datetime(2026, 3, 21),
        spot_prices={"SPY": 520.0},
        as_of_date=date(2026, 3, 21),
    )
    assert account.positions[option_id.key()].quantity == 0
    # Settlement pays (520-500)*100 = 2000 on top of prior cash.
    assert account.cash == 10_000 - 501 + 2000
