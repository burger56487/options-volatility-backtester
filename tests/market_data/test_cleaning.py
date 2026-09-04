from datetime import date

from src.market_data.cleaning import (
    deduplicate_underlying_bars,
    find_duplicate_underlying_bars,
)
from src.market_data.schemas import UnderlyingBar


def _make_bar() -> UnderlyingBar:
    return UnderlyingBar(
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


def test_duplicate_underlying_bars_are_detected():
    bar = _make_bar()
    duplicates = find_duplicate_underlying_bars([bar, bar])
    assert len(duplicates) == 1
    assert next(iter(duplicates.values())) == 2


def test_duplicate_underlying_bars_are_removed():
    bar = _make_bar()
    result = deduplicate_underlying_bars([bar, bar])
    assert len(result) == 1
