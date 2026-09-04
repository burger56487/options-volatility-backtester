from datetime import date, datetime

from src.market_data.schemas import (
    DataType,
    OptionQuote,
    OptionType,
    UnderlyingBar,
)
from src.market_data.validators import (
    validate_option_quote,
    validate_underlying_bar,
)


def make_valid_option_quote() -> OptionQuote:
    return OptionQuote(
        timestamp=datetime(2026, 1, 2, 16, 0),
        underlying_symbol="SPY",
        expiry=date(2026, 2, 20),
        strike=100.0,
        option_type=OptionType.CALL,
        bid=5.0,
        ask=5.2,
        spot=102.0,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        source="synthetic_generator",
        data_type=DataType.SYNTHETIC,
    )


def test_valid_underlying_bar():
    bar = UnderlyingBar(
        trade_date=date(2026, 1, 2),
        symbol="SPY",
        open=100.0,
        high=103.0,
        low=99.0,
        close=102.0,
        adjusted_close=102.0,
        volume=1_000_000,
        source="public_market_data",
    )
    result = validate_underlying_bar(bar)
    assert result.valid
    assert result.errors == []


def test_invalid_ohlc_is_rejected():
    bar = UnderlyingBar(
        trade_date=date(2026, 1, 2),
        symbol="SPY",
        open=100.0,
        high=99.0,
        low=98.0,
        close=102.0,
        adjusted_close=102.0,
        volume=1_000_000,
        source="public_market_data",
    )
    result = validate_underlying_bar(bar)
    assert not result.valid
    assert any(
        issue.code == "INVALID_HIGH"
        for issue in result.errors
    )


def test_crossed_option_market_is_rejected():
    quote = make_valid_option_quote()
    invalid_quote = OptionQuote(
        **{
            **quote.__dict__,
            "bid": 5.5,
            "ask": 5.2,
        }
    )
    result = validate_option_quote(invalid_quote)
    assert not result.valid
    assert any(
        issue.code == "CROSSED_MARKET"
        for issue in result.errors
    )


def test_expired_option_is_rejected():
    quote = make_valid_option_quote()
    invalid_quote = OptionQuote(
        **{
            **quote.__dict__,
            "expiry": date(2026, 1, 2),
        }
    )
    result = validate_option_quote(invalid_quote)
    assert not result.valid
    assert any(
        issue.code == "EXPIRED_OPTION"
        for issue in result.errors
    )


def test_wide_spread_produces_warning():
    quote = make_valid_option_quote()
    warning_quote = OptionQuote(
        **{
            **quote.__dict__,
            "bid": 1.0,
            "ask": 3.0,
        }
    )
    result = validate_option_quote(
        warning_quote,
        max_relative_spread=0.50,
    )
    assert any(
        issue.code == "WIDE_SPREAD"
        for issue in result.warnings
    )
