"""Duplicate detection and removal for schema records."""

from __future__ import annotations

from collections import Counter

from .schemas import OptionQuote, UnderlyingBar


def underlying_key(bar: UnderlyingBar) -> tuple:
    return (bar.symbol, bar.trade_date)


def option_quote_key(quote: OptionQuote) -> tuple:
    return (
        quote.timestamp,
        quote.underlying_symbol,
        quote.expiry,
        quote.strike,
        quote.option_type.value,
    )


def find_duplicate_underlying_bars(
    bars: list[UnderlyingBar],
) -> dict[tuple, int]:
    counts = Counter(underlying_key(bar) for bar in bars)
    return {
        key: count
        for key, count in counts.items()
        if count > 1
    }


def find_duplicate_option_quotes(
    quotes: list[OptionQuote],
) -> dict[tuple, int]:
    counts = Counter(option_quote_key(quote) for quote in quotes)
    return {
        key: count
        for key, count in counts.items()
        if count > 1
    }


def deduplicate_underlying_bars(
    bars: list[UnderlyingBar],
) -> list[UnderlyingBar]:
    unique = {}
    for bar in bars:
        unique[underlying_key(bar)] = bar
    return sorted(
        unique.values(),
        key=lambda item: (item.symbol, item.trade_date),
    )


def deduplicate_option_quotes(
    quotes: list[OptionQuote],
) -> list[OptionQuote]:
    unique = {}
    for quote in quotes:
        unique[option_quote_key(quote)] = quote
    return sorted(
        unique.values(),
        key=lambda item: (
            item.timestamp,
            item.underlying_symbol,
            item.expiry,
            item.strike,
            item.option_type.value,
        ),
    )
