"""Tests for multi-contract Greeks-aware quoting."""

from __future__ import annotations

import pytest

from src.market_making.greeks_book import (
    BookContract,
    GreeksQuoteConfig,
    expiry_bucket_id,
    quote_multi_contract_book,
)


def _contract(
    key: str,
    strike: float = 100.0,
    tau: float = 0.05,
    option_type: str = "call",
    quantity: float = 1.0,
    multiplier: float = 1.0,
) -> BookContract:
    return BookContract(
        key=key,
        strike=strike,
        tau=tau,
        option_type=option_type,
        quantity=quantity,
        multiplier=multiplier,
    )


def test_empty_book_has_no_quotes_and_zero_net_greeks():
    result = quote_multi_contract_book(
        spot=100.0,
        contracts=[],
        config=GreeksQuoteConfig(),
    )
    assert result.quotes == ()
    assert result.net_delta == 0.0
    assert result.net_gamma == 0.0
    assert result.net_vega == 0.0


def test_zero_quantity_book_quotes_symmetrically():
    contract = _contract("c1", quantity=0.0)
    result = quote_multi_contract_book(
        spot=100.0,
        contracts=[contract],
        config=GreeksQuoteConfig(base_half_spread=0.05),
    )
    (quote,) = result.quotes
    assert quote.bid_offset == pytest.approx(-quote.ask_offset)
    assert quote.bid_offset == pytest.approx(-0.05)
    assert quote.bid < quote.mid < quote.ask


def test_long_call_inventory_lowers_both_quotes():
    config = GreeksQuoteConfig(
        base_half_spread=0.05,
        volatility=0.2,
        risk_aversion=1.0,
    )
    flat = quote_multi_contract_book(
        spot=100.0,
        contracts=[_contract("c1", quantity=0.0)],
        config=config,
    ).quotes[0]
    long_call = quote_multi_contract_book(
        spot=100.0,
        contracts=[_contract("c1", quantity=1.0)],
        config=config,
    ).quotes[0]
    assert long_call.ask_offset < flat.ask_offset
    assert long_call.bid_offset < flat.bid_offset
    assert long_call.ask_offset > long_call.bid_offset


def test_short_put_inventory_raises_quotes():
    """A short put adds positive delta, so quotes move up to encourage buys."""
    config = GreeksQuoteConfig(base_half_spread=0.05)
    flat = quote_multi_contract_book(
        spot=100.0,
        contracts=[
            _contract("p1", option_type="put", quantity=0.0)
        ],
        config=config,
    ).quotes[0]
    short_put = quote_multi_contract_book(
        spot=100.0,
        contracts=[
            _contract("p1", option_type="put", quantity=-1.0)
        ],
        config=config,
    ).quotes[0]
    assert short_put.bid_offset > flat.bid_offset
    assert short_put.ask_offset > flat.ask_offset


def test_long_put_negative_delta_shifts_put_quotes_down():
    """Long puts create negative net delta; the option-level shift uses the
    put's negative delta sign to push its quotes down (sell bias)."""
    config = GreeksQuoteConfig(base_half_spread=0.05)
    result = quote_multi_contract_book(
        spot=100.0,
        contracts=[
            _contract("p1", option_type="put", quantity=0.0)
        ],
        config=config,
    )
    flat = result.quotes[0]
    long_result = quote_multi_contract_book(
        spot=100.0,
        contracts=[
            _contract("p1", option_type="put", quantity=1.0)
        ],
        config=config,
    )
    long_put = long_result.quotes[0]
    assert long_result.net_delta < 0.0
    assert long_put.ask_offset < flat.ask_offset


def test_gamma_risk_widens_spread():
    config = GreeksQuoteConfig(
        base_half_spread=0.05,
        gamma_charge=0.5,
        gamma_scale=1.0,
    )
    one = quote_multi_contract_book(
        spot=100.0,
        contracts=[_contract("c1", quantity=1.0)],
        config=config,
    ).quotes[0]
    two = quote_multi_contract_book(
        spot=100.0,
        contracts=[_contract("c1", quantity=2.0)],
        config=config,
    ).quotes[0]
    half_spread_one = 0.5 * (one.ask - one.bid)
    half_spread_two = 0.5 * (two.ask - two.bid)
    assert half_spread_two > half_spread_one


def test_net_greeks_aggregate_across_strikes_and_expiries():
    config = GreeksQuoteConfig()
    result = quote_multi_contract_book(
        spot=100.0,
        contracts=[
            _contract("itm-call", strike=90.0, tau=0.10),
            _contract("otm-call", strike=110.0, tau=0.20),
            _contract(
                "short-itm-put",
                strike=110.0,
                tau=0.10,
                option_type="put",
                quantity=-1.0,
            ),
        ],
        config=config,
    )
    assert result.net_delta > 0.0
    assert result.net_gamma > 0.0
    assert result.net_vega > 0.0
    assert len(result.quotes) == 3


def test_quote_prices_use_bsm_mid_with_offsets():
    config = GreeksQuoteConfig(base_half_spread=0.10)
    quote = quote_multi_contract_book(
        spot=100.0,
        contracts=[_contract("c1", strike=100.0, tau=0.1, quantity=0.0)],
        config=config,
    ).quotes[0]
    assert 0.0 < quote.mid < 100.0
    assert quote.ask - quote.mid == pytest.approx(0.10)
    assert quote.mid - quote.bid == pytest.approx(0.10)


def test_expiry_bucket_mapping():
    assert expiry_bucket_id(1.0 / 252.0) == "0-7d"
    assert expiry_bucket_id(20.0 / 252.0) == "7-30d"
    assert expiry_bucket_id(60.0 / 252.0) == "30-90d"
    assert expiry_bucket_id(120.0 / 252.0) == "90d+"


def test_bucket_summary_groups_contracts_and_reports_risk():
    config = GreeksQuoteConfig()
    result = quote_multi_contract_book(
        spot=100.0,
        contracts=[
            _contract("short-tenor", tau=5.0 / 252.0),
            _contract("mid-tenor", tau=20.0 / 252.0),
            _contract("mid-tenor-2", strike=105.0, tau=25.0 / 252.0),
        ],
        config=config,
    )
    buckets = {row.bucket: row for row in result.bucket_summary}
    assert set(buckets) == {"0-7d", "7-30d"}
    assert buckets["0-7d"].n_contracts == 1
    assert buckets["7-30d"].n_contracts == 2
    assert buckets["7-30d"].net_gamma > 0.0


def test_offsetting_gamma_book_is_narrower_than_aligned_book():
    """Net gamma cancels across legs, so risk widening shrinks."""
    config = GreeksQuoteConfig(
        base_half_spread=0.05,
        gamma_charge=1.0,
        gamma_scale=1.0,
    )
    aligned = quote_multi_contract_book(
        spot=100.0,
        contracts=[
            _contract("c1", strike=100.0, quantity=1.0),
            _contract("c2", strike=105.0, quantity=1.0),
        ],
        config=config,
    )
    offset = quote_multi_contract_book(
        spot=100.0,
        contracts=[
            _contract("c1", strike=100.0, quantity=1.0),
            _contract("c2", strike=105.0, quantity=-1.0),
        ],
        config=config,
    )
    aligned_width = 0.5 * (aligned.quotes[0].ask - aligned.quotes[0].bid)
    offset_width = 0.5 * (offset.quotes[0].ask - offset.quotes[0].bid)
    assert offset_width < aligned_width
