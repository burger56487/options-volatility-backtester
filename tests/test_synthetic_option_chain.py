import pandas as pd
import pytest

from src.market_data.synthetic_option_chain import (
    VolatilitySurfaceParameters,
    blended_realised_volatility,
    create_synthetic_option_chain,
    select_atm_straddle,
    synthetic_implied_volatility,
)


def test_blended_realised_volatility_uses_default_weights():
    result = blended_realised_volatility(
        realized_vol_20d=0.20,
        realized_vol_60d=0.15,
        realized_vol_252d=0.10,
    )

    expected = 0.5 * 0.20 + 0.3 * 0.15 + 0.2 * 0.10

    assert result == pytest.approx(expected)


def test_blended_volatility_rejects_weights_not_summing_to_one():
    with pytest.raises(
        ValueError,
        match="weights must sum to 1.0",
    ):
        blended_realised_volatility(
            realized_vol_20d=0.20,
            realized_vol_60d=0.15,
            realized_vol_252d=0.10,
            weights=(0.5, 0.5, 0.5),
        )


def test_lower_strike_has_higher_volatility_under_negative_skew():
    parameters = VolatilitySurfaceParameters(
        put_skew=-0.10,
    )

    low_strike_volatility = synthetic_implied_volatility(
        spot=100.0,
        strike=90.0,
        time_to_expiry=30 / 365,
        base_volatility=0.20,
        option_type="put",
        parameters=parameters,
    )

    high_strike_volatility = synthetic_implied_volatility(
        spot=100.0,
        strike=110.0,
        time_to_expiry=30 / 365,
        base_volatility=0.20,
        option_type="put",
        parameters=parameters,
    )

    assert low_strike_volatility > high_strike_volatility


def test_longer_expiry_has_higher_volatility_with_positive_term_slope():
    parameters = VolatilitySurfaceParameters(
        term_structure_slope=0.03,
    )

    short_term_volatility = synthetic_implied_volatility(
        spot=100.0,
        strike=100.0,
        time_to_expiry=30 / 365,
        base_volatility=0.20,
        option_type="call",
        parameters=parameters,
    )

    long_term_volatility = synthetic_implied_volatility(
        spot=100.0,
        strike=100.0,
        time_to_expiry=180 / 365,
        base_volatility=0.20,
        option_type="call",
        parameters=parameters,
    )

    assert long_term_volatility > short_term_volatility


def test_option_chain_has_expected_number_of_rows():
    chain = create_synthetic_option_chain(
        valuation_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        base_volatility=0.20,
        risk_free_rate=0.03,
        days_to_expiry=(30, 60),
        strike_multipliers=(0.9, 1.0, 1.1),
    )

    assert len(chain) == 2 * 3 * 2


def test_option_chain_has_valid_bid_ask_quotes_and_greeks():
    chain = create_synthetic_option_chain(
        valuation_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        base_volatility=0.20,
        risk_free_rate=0.03,
        days_to_expiry=(30,),
        strike_multipliers=(0.9, 1.0, 1.1),
    )

    assert (chain["bid"] >= 0).all()
    assert (chain["ask"] >= chain["bid"]).all()
    assert (chain["mid"] >= chain["bid"]).all()
    assert (chain["mid"] <= chain["ask"]).all()
    assert (chain["implied_volatility"] > 0).all()
    assert (chain["gamma"] > 0).all()
    assert (chain["vega"] > 0).all()


def test_option_chain_put_delta_is_negative_and_call_delta_is_positive():
    chain = create_synthetic_option_chain(
        valuation_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        base_volatility=0.20,
        risk_free_rate=0.03,
        days_to_expiry=(30,),
        strike_multipliers=(1.0,),
    )

    call_delta = chain.loc[
        chain["option_type"] == "call",
        "delta",
    ].iloc[0]

    put_delta = chain.loc[
        chain["option_type"] == "put",
        "delta",
    ].iloc[0]

    assert call_delta > 0
    assert put_delta < 0


def test_select_atm_straddle_returns_call_and_put():
    chain = create_synthetic_option_chain(
        valuation_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        base_volatility=0.20,
        risk_free_rate=0.03,
        days_to_expiry=(30, 60),
        strike_multipliers=(0.9, 1.0, 1.1),
    )

    straddle = select_atm_straddle(
        chain=chain,
        days_to_expiry=30,
    )

    assert len(straddle) == 2
    assert set(straddle["option_type"]) == {"call", "put"}
    assert (straddle["strike"] == 100.0).all()


def test_select_atm_straddle_rejects_unavailable_maturity():
    chain = create_synthetic_option_chain(
        valuation_date=pd.Timestamp("2025-01-02"),
        spot=100.0,
        base_volatility=0.20,
        risk_free_rate=0.03,
        days_to_expiry=(30,),
        strike_multipliers=(1.0,),
    )

    with pytest.raises(
        ValueError,
        match="No options found",
    ):
        select_atm_straddle(
            chain=chain,
            days_to_expiry=60,
        )
