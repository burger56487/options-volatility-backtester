from datetime import date, datetime

from src.portfolio.account import Account
from src.portfolio.fills import Fill
from src.portfolio.identifiers import (
    InstrumentId,
    InstrumentType,
    OptionType,
)
from src.portfolio.orders import Side
from src.portfolio.reconciliation import reconcile_pnl_bridge


def test_pnl_bridge_passes_after_roundtrip():
    account = Account(
        account_id="a1",
        run_id="r1",
        base_currency="USD",
        initial_capital=100_000.0,
    )
    option_id = InstrumentId(
        instrument_type=InstrumentType.OPTION,
        symbol="SPY",
        expiry=date(2026, 3, 20),
        strike=500.0,
        option_type=OptionType.CALL,
    )
    key = option_id.key()
    for side, price in [(Side.BUY, 5.0), (Side.SELL, 7.0)]:
        account.apply_fill(
            Fill(
                fill_id=f"f{side.value}",
                order_id="o1",
                run_id="r1",
                timestamp=datetime(2026, 1, 2),
                instrument_id=option_id,
                side=side,
                quantity=1,
                price=price,
                multiplier=100.0,
                commission=1.0,
            )
        )
    result = reconcile_pnl_bridge(account, market_prices={key: 7.0})
    assert result.passed
    assert result.difference == 0.0
