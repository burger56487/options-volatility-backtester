from datetime import date, datetime

from src.portfolio.fills import Fill
from src.portfolio.identifiers import (
    InstrumentId,
    InstrumentType,
    OptionType,
)
from src.portfolio.orders import Side
from src.portfolio.positions import Position


def _option_id():
    return InstrumentId(
        instrument_type=InstrumentType.OPTION,
        symbol="SPY",
        expiry=date(2026, 3, 20),
        strike=500.0,
        option_type=OptionType.CALL,
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


def test_close_long_position_calculates_realised_pnl():
    position = Position(instrument_id=_option_id(), multiplier=100.0)
    position.apply_fill(_fill(Side.BUY, 1, 5.0))
    position.apply_fill(_fill(Side.SELL, 1, 7.0))
    assert position.quantity == 0
    assert position.realised_pnl == 200.0


def test_position_can_reverse_from_long_to_short():
    position = Position(instrument_id=_option_id(), multiplier=100.0)
    position.apply_fill(_fill(Side.BUY, 2, 5.0))
    position.apply_fill(_fill(Side.SELL, 3, 7.0))
    assert position.quantity == -1
    assert position.realised_pnl == 400.0
    assert position.average_cost == 7.0


def test_short_average_cost_after_adding():
    position = Position(instrument_id=_option_id(), multiplier=100.0)
    position.apply_fill(_fill(Side.SELL, 1, 5.0))
    position.apply_fill(_fill(Side.SELL, 1, 7.0))
    assert position.quantity == -2
    assert position.average_cost == 6.0
