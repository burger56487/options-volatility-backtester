from datetime import date

from src.strategy_lib.strategies import (
    iron_butterfly,
    long_straddle,
    long_strangle,
    price_strategy,
)


def test_long_straddle_premium_and_vega_positive():
    output = price_strategy(
        long_straddle(),
        spot=100.0,
        valuation_date=date(2026, 9, 4),
        risk_free_rate=0.04,
    )
    assert output.premium > 0
    assert output.vega > 0
    assert output.theta < 0


def test_long_strangle_delta_near_zero():
    output = price_strategy(
        long_strangle(),
        spot=100.0,
        valuation_date=date(2026, 9, 4),
        risk_free_rate=0.04,
    )
    assert abs(output.delta) < 20.0


def test_iron_butterfly_is_short_vega():
    output = price_strategy(
        iron_butterfly(),
        spot=100.0,
        valuation_date=date(2026, 9, 4),
        risk_free_rate=0.04,
    )
    assert output.vega < 0
    assert output.premium < 0  # credit structure: cash received upfront
    assert output.theta > 0
