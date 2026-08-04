import pandas as pd
import pytest

from src.strategy.delta_hedging import (
    DeltaHedger,
    UnderlyingTransactionCostModel,
)


def test_target_position_offsets_option_delta():
    assert DeltaHedger.target_position(25.0) == -25.0
    assert DeltaHedger.target_position(-12.5) == 12.5


def test_target_position_rounds_without_fractional_shares():
    target = DeltaHedger.target_position(
        option_delta=12.6,
        allow_fractional_shares=False,
    )

    assert target == -13.0


def test_transaction_cost_includes_commission_and_slippage():
    model = UnderlyingTransactionCostModel(
        commission_per_share=0.01,
        slippage_bps=2.0,
    )

    cost = model.transaction_cost(
        quantity=100.0,
        spot=50.0,
    )

    expected_commission = 100.0 * 0.01
    expected_slippage = 100.0 * 50.0 * 2.0 / 10_000.0

    assert cost == pytest.approx(
        expected_commission + expected_slippage
    )


def test_rebalance_creates_short_hedge_for_positive_option_delta():
    hedger = DeltaHedger(
        cost_model=UnderlyingTransactionCostModel(
            slippage_bps=0.0
        )
    )

    trade = hedger.rebalance(
        trade_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        option_delta=20.0,
    )

    assert trade is not None
    assert trade.quantity == pytest.approx(-20.0)
    assert hedger.position == pytest.approx(-20.0)
    assert trade.post_hedge_delta == pytest.approx(0.0)


def test_rebalance_creates_long_hedge_for_negative_option_delta():
    hedger = DeltaHedger(
        cost_model=UnderlyingTransactionCostModel(
            slippage_bps=0.0
        )
    )

    trade = hedger.rebalance(
        trade_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        option_delta=-15.0,
    )

    assert trade is not None
    assert trade.quantity == pytest.approx(15.0)
    assert hedger.position == pytest.approx(15.0)
    assert trade.post_hedge_delta == pytest.approx(0.0)


def test_delta_threshold_prevents_small_rebalance():
    hedger = DeltaHedger(
        delta_threshold=5.0,
        cost_model=UnderlyingTransactionCostModel(
            slippage_bps=0.0
        ),
    )

    first_trade = hedger.rebalance(
        trade_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        option_delta=20.0,
    )

    second_trade = hedger.rebalance(
        trade_date=pd.Timestamp("2025-01-03"),
        spot=101.0,
        option_delta=16.0,
    )

    assert first_trade is not None
    assert second_trade is None
    assert hedger.position == pytest.approx(-20.0)


def test_adverse_execution_price_for_buy_and_sell():
    hedger = DeltaHedger(
        cost_model=UnderlyingTransactionCostModel(
            slippage_bps=10.0
        )
    )

    buy_price = hedger._execution_price(
        spot=100.0,
        quantity=10.0,
        slippage_bps=10.0,
    )

    sell_price = hedger._execution_price(
        spot=100.0,
        quantity=-10.0,
        slippage_bps=10.0,
    )

    assert buy_price == pytest.approx(100.10)
    assert sell_price == pytest.approx(99.90)


def test_hedge_equity_changes_with_spot_price():
    hedger = DeltaHedger(
        cost_model=UnderlyingTransactionCostModel(
            slippage_bps=0.0
        )
    )

    hedger.rebalance(
        trade_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        option_delta=-10.0,
    )

    initial_equity = hedger.total_equity(spot=100.0)
    later_equity = hedger.total_equity(spot=110.0)

    assert initial_equity == pytest.approx(0.0)
    assert later_equity == pytest.approx(100.0)


def test_turnover_ratio():
    hedger = DeltaHedger(
        cost_model=UnderlyingTransactionCostModel(
            slippage_bps=0.0
        )
    )

    hedger.rebalance(
        trade_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        option_delta=-10.0,
    )

    assert hedger.turnover_ratio(
        initial_portfolio_value=1_000.0
    ) == pytest.approx(1.0)


def test_trade_log_has_expected_columns():
    hedger = DeltaHedger()

    hedger.rebalance(
        trade_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        option_delta=5.0,
    )

    trade_log = hedger.trade_log()

    assert len(trade_log) == 1
    assert "transaction_cost" in trade_log.columns
    assert "post_hedge_delta" in trade_log.columns


def test_negative_threshold_raises_error():
    with pytest.raises(
        ValueError,
        match="delta_threshold must be non-negative",
    ):
        DeltaHedger(delta_threshold=-1.0)
