from datetime import date, datetime

from src.portfolio.account import Account
from src.portfolio.fills import Fill
from src.portfolio.identifiers import (
    InstrumentId,
    InstrumentType,
)
from src.portfolio.orders import Side
from src.portfolio.valuation import MarketSnapshot
from src.risk.limits import RiskLimits
from src.risk.pre_trade import (
    max_allowed_fill_quantity,
    simulate_post_trade_limit_check,
)


def _stock_id():
    return InstrumentId(
        instrument_type=InstrumentType.STOCK,
        symbol="SPY",
    )


def test_leverage_limit_rejects_oversized_order():
    account = Account(
        account_id="a1",
        run_id="r1",
        base_currency="USD",
        initial_capital=100_000.0,
    )
    key = _stock_id().key()
    limits = RiskLimits(
        max_gross_exposure=150_000.0,
        max_leverage=1.5,
        max_abs_delta=1e9,
        max_abs_gamma=1e9,
        max_abs_vega=1e9,
        max_daily_loss=1e9,
        max_drawdown=1.0,
        min_cash_buffer=0.0,
    )
    market = MarketSnapshot(
        timestamp=datetime(2026, 1, 2),
        prices={key: 100.0},
        deltas={key: 1.0},
        gammas={key: 0.0},
        vegas={key: 0.0},
        thetas={key: 0.0},
        rhos={key: 0.0},
    )

    def make_fill(quantity):
        return Fill(
            fill_id="f",
            order_id="o",
            run_id="r1",
            timestamp=datetime(2026, 1, 2),
            instrument_id=_stock_id(),
            side=Side.BUY,
            quantity=quantity,
            price=100.0,
            multiplier=1.0,
        )

    full_check = simulate_post_trade_limit_check(
        account,
        make_fill(2_000),
        market,
        limits,
        daily_pnl=0.0,
    )
    assert not full_check.allowed
    max_qty = max_allowed_fill_quantity(
        account,
        2_000,
        make_fill,
        market,
        limits,
        daily_pnl=0.0,
    )
    assert 0 < max_qty < 2_000
