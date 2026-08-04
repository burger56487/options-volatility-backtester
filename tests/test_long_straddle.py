import pandas as pd
import pytest

from src.strategy.long_straddle import (
    LongStraddle,
    OptionContract,
    OptionPosition,
    build_long_atm_straddle,
)


def create_option_contract(
    option_type: str,
) -> OptionContract:
    return OptionContract(
        option_type=option_type,  # type: ignore
        strike=100.0,
        expiry_date=pd.Timestamp("2025-02-01"),
        multiplier=100,
    )


def create_straddle() -> LongStraddle:
    call_position = OptionPosition(
        contract=create_option_contract("call"),
        quantity=1,
        entry_price=4.00,
    )

    put_position = OptionPosition(
        contract=create_option_contract("put"),
        quantity=1,
        entry_price=3.50,
    )

    return LongStraddle(
        call_position=call_position,
        put_position=put_position,
    )


def test_option_position_entry_notional_for_long_option():
    position = OptionPosition(
        contract=create_option_contract("call"),
        quantity=2,
        entry_price=4.0,
    )

    assert position.entry_notional == pytest.approx(-800.0)


def test_option_intrinsic_value():
    call_position = OptionPosition(
        contract=create_option_contract("call"),
        quantity=1,
        entry_price=4.0,
    )

    put_position = OptionPosition(
        contract=create_option_contract("put"),
        quantity=1,
        entry_price=3.5,
    )

    assert call_position.intrinsic_value(110.0) == 10.0
    assert call_position.intrinsic_value(95.0) == 0.0
    assert put_position.intrinsic_value(90.0) == 10.0
    assert put_position.intrinsic_value(105.0) == 0.0


def test_option_value_at_expiry_is_intrinsic_value():
    position = OptionPosition(
        contract=create_option_contract("call"),
        quantity=1,
        entry_price=4.0,
    )

    value = position.market_value(
        valuation_date=pd.Timestamp("2025-02-01"),
        spot=110.0,
        risk_free_rate=0.03,
        volatility=0.20,
    )

    assert value == pytest.approx(1000.0)


def test_long_straddle_entry_cost():
    straddle = create_straddle()

    assert straddle.entry_cost == pytest.approx(750.0)


def test_long_straddle_combined_delta_is_close_to_zero_at_atm():
    straddle = create_straddle()

    greeks = straddle.combined_greeks(
        valuation_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        risk_free_rate=0.03,
        call_volatility=0.20,
        put_volatility=0.20,
    )

    assert abs(greeks["delta"]) < 10.0
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0
    assert greeks["theta"] < 0


def test_long_straddle_value_increases_after_large_price_move():
    straddle = create_straddle()

    initial_value = straddle.market_value(
        valuation_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        risk_free_rate=0.03,
        call_volatility=0.20,
        put_volatility=0.20,
    )

    moved_value = straddle.market_value(
        valuation_date=pd.Timestamp("2025-01-15"),
        spot=115.0,
        risk_free_rate=0.03,
        call_volatility=0.20,
        put_volatility=0.20,
    )

    assert moved_value > initial_value


def test_long_straddle_rejects_mismatched_strikes():
    call_position = OptionPosition(
        contract=OptionContract(
            option_type="call",
            strike=100.0,
            expiry_date=pd.Timestamp("2025-02-01"),
        ),
        quantity=1,
        entry_price=4.0,
    )

    put_position = OptionPosition(
        contract=OptionContract(
            option_type="put",
            strike=105.0,
            expiry_date=pd.Timestamp("2025-02-01"),
        ),
        quantity=1,
        entry_price=3.5,
    )

    with pytest.raises(
        ValueError,
        match="strikes must match",
    ):
        LongStraddle(
            call_position=call_position,
            put_position=put_position,
        )


def test_build_long_atm_straddle_uses_ask_prices():
    chain = pd.DataFrame(
        {
            "valuation_date": [
                pd.Timestamp("2025-01-02"),
                pd.Timestamp("2025-01-02"),
            ],
            "expiry_date": [
                pd.Timestamp("2025-02-01"),
                pd.Timestamp("2025-02-01"),
            ],
            "option_type": ["call", "put"],
            "strike": [100.0, 100.0],
            "bid": [3.9, 3.4],
            "ask": [4.0, 3.5],
        }
    )

    straddle = build_long_atm_straddle(
        chain=chain,
        quantity=2,
    )

    assert straddle.quantity == 2
    assert straddle.entry_cost == pytest.approx(
        (4.0 + 3.5) * 2 * 100
    )


def test_contract_rejects_invalid_option_type():
    with pytest.raises(
        ValueError,
        match="option_type must be either",
    ):
        OptionContract(
            option_type="straddle",  # type: ignore
            strike=100.0,
            expiry_date=pd.Timestamp("2025-02-01"),
        )


def test_position_rejects_zero_quantity():
    with pytest.raises(
        ValueError,
        match="quantity must not be zero",
    ):
        OptionPosition(
            contract=create_option_contract("call"),
            quantity=0,
            entry_price=4.0,
        )
