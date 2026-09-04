from src.financing.margin import (
    estimate_long_option_margin,
    estimate_short_option_margin,
)


def test_long_option_margin_equals_premium():
    estimate = estimate_long_option_margin(500.0)
    assert estimate.initial_margin == 500.0
    assert estimate.maintenance_margin == 500.0


def test_short_option_margin_is_positive():
    estimate = estimate_short_option_margin(
        option_market_value=-500.0,
        underlying_market_value=50_000.0,
        out_of_the_money_amount=300.0,
    )
    assert estimate.initial_margin > 500.0
    assert estimate.maintenance_margin < estimate.initial_margin
