"""Multi-contract Greeks-aware quoting across strikes and expiries.

Quotes are computed per option contract around the BSM fair mid.  The
portfolio-level reservation-price shift uses the net delta exposure in the
Avellaneda-Stoikov form, translated to each contract through its own delta;
gamma and vega risk widen the per-contract half-spread in proportion to the
contract's share of the net book risk.  Contracts are additionally grouped
into expiry buckets so risk can be viewed (and later managed) by tenor.

Greek conventions match the account engine: per-share values from
``price_and_greeks`` are scaled by the contract multiplier for exposures, and
option premium quotes are per-share prices.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.pricing.black_scholes import price_and_greeks


@dataclass(frozen=True)
class BookContract:
    """One option leg held by the market maker."""

    key: str
    strike: float
    tau: float  # time to expiry in years
    option_type: str  # "call" or "put"
    quantity: float  # signed contracts
    multiplier: float = 1.0


@dataclass(frozen=True)
class GreeksQuoteConfig:
    base_half_spread: float = 0.05
    min_half_spread: float = 0.0
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0
    volatility: float = 0.2
    risk_aversion: float = 1.0
    gamma_charge: float = 0.5
    vega_charge: float = 0.02
    gamma_scale: float = 100.0
    vega_scale: float = 100.0
    max_spread_multiple: float = 4.0


@dataclass(frozen=True)
class GreeksQuote:
    key: str
    strike: float
    tau: float
    option_type: str
    bucket: str
    mid: float
    bid: float
    ask: float
    bid_offset: float
    ask_offset: float
    half_spread: float
    delta: float  # per share
    gamma: float  # per share
    vega: float  # per share
    theta: float  # per share, per year
    rho: float  # per share, per 1.0 rate change


@dataclass(frozen=True)
class ExpiryBucketSummary:
    bucket: str
    n_contracts: int
    net_delta: float
    net_gamma: float
    net_vega: float
    mean_half_spread: float


@dataclass(frozen=True)
class MultiQuoteResult:
    quotes: tuple[GreeksQuote, ...]
    net_delta: float
    net_gamma: float
    net_vega: float
    bucket_summary: tuple[ExpiryBucketSummary, ...]


_BUCKET_ORDER = ["0-7d", "7-30d", "30-90d", "90d+"]


def expiry_bucket_id(
    tau_years: float,
    boundaries_days: tuple[float, ...] = (7.0, 30.0, 90.0),
) -> str:
    """Map years-to-expiry to a bucket label.

    One year is treated as 252 trading days, matching the account engine's
    time convention used elsewhere in the repository.
    """
    days = tau_years * 252.0
    previous = 0.0
    for boundary in boundaries_days:
        if days <= boundary:
            return f"{int(previous)}-{int(boundary)}d"
        previous = boundary
    return "90d+"


def _validate(spot: float, contracts: list[BookContract], config: GreeksQuoteConfig) -> None:
    if spot <= 0.0:
        raise ValueError("spot must be positive.")
    if config.base_half_spread < 0.0:
        raise ValueError("base_half_spread must be non-negative.")
    if config.volatility <= 0.0:
        raise ValueError("volatility must be positive.")
    for contract in contracts:
        if contract.tau <= 0.0:
            raise ValueError(f"contract {contract.key} has non-positive tau.")
        if contract.strike <= 0.0:
            raise ValueError(f"contract {contract.key} has non-positive strike.")
        if contract.option_type not in {"call", "put"}:
            raise ValueError(
                f"contract {contract.key} option_type must be call or put."
            )


def quote_multi_contract_book(
    spot: float,
    contracts: list[BookContract],
    config: GreeksQuoteConfig = GreeksQuoteConfig(),
) -> MultiQuoteResult:
    """Quote every listed contract around its BSM mid with Greeks-aware offsets."""
    _validate(spot, contracts, config)
    if not contracts:
        return MultiQuoteResult(
            quotes=(),
            net_delta=0.0,
            net_gamma=0.0,
            net_vega=0.0,
            bucket_summary=(),
        )

    priced = []
    for contract in contracts:
        result = price_and_greeks(
            spot=spot,
            strike=contract.strike,
            time_to_expiry=contract.tau,
            risk_free_rate=config.risk_free_rate,
            volatility=config.volatility,
            option_type=contract.option_type,
            dividend_yield=config.dividend_yield,
        )
        multiplier = float(contract.multiplier)
        priced.append(
            {
                "contract": contract,
                "price": result.price,
                "delta": result.delta,
                "gamma": result.gamma,
                "vega": result.vega,
                "theta": result.theta,
                "rho": result.rho,
                "delta_exposure": result.delta
                * multiplier
                * float(contract.quantity),
                "gamma_exposure": result.gamma
                * multiplier
                * float(contract.quantity),
                "vega_exposure": result.vega
                * multiplier
                * float(contract.quantity),
            }
        )

    net_delta = sum(row["delta_exposure"] for row in priced)
    net_gamma = sum(row["gamma_exposure"] for row in priced)
    net_vega = sum(row["vega_exposure"] for row in priced)

    abs_gamma_sum = sum(abs(row["gamma_exposure"]) for row in priced)
    abs_vega_sum = sum(abs(row["vega_exposure"]) for row in priced)
    gamma_weight = {
        id(row): (
            abs(row["gamma_exposure"]) / abs_gamma_sum
            if abs_gamma_sum > 0.0
            else 0.0
        )
        for row in priced
    }
    vega_weight = {
        id(row): (
            abs(row["vega_exposure"]) / abs_vega_sum
            if abs_vega_sum > 0.0
            else 0.0
        )
        for row in priced
    }

    absolute_volume = sum(
        abs(row["contract"].quantity) * row["contract"].multiplier
        for row in priced
    )
    reference_tau = (
        sum(
            abs(row["contract"].quantity)
            * row["contract"].multiplier
            * row["contract"].tau
            for row in priced
        )
        / absolute_volume
        if absolute_volume > 0.0
        else 0.0
    )
    reservation_shift = (
        -config.risk_aversion
        * config.volatility**2
        * reference_tau
        * net_delta
    )

    quotes = []
    for row in priced:
        contract = row["contract"]
        gamma_multiple = (
            config.gamma_charge
            * (abs(net_gamma) / config.gamma_scale)
            * gamma_weight[id(row)]
        )
        vega_multiple = (
            config.vega_charge
            * (abs(net_vega) / config.vega_scale)
            * vega_weight[id(row)]
        )
        width_multiple = 1.0 + gamma_multiple + vega_multiple
        capped = min(width_multiple, config.max_spread_multiple)
        half_spread = max(
            config.base_half_spread * capped,
            config.min_half_spread,
        )
        shift = reservation_shift * row["delta"]
        bid_offset = shift - half_spread
        ask_offset = shift + half_spread
        mid = float(row["price"])
        quotes.append(
            GreeksQuote(
                key=contract.key,
                strike=contract.strike,
                tau=contract.tau,
                option_type=contract.option_type,
                bucket=expiry_bucket_id(contract.tau),
                mid=mid,
                bid=mid + bid_offset,
                ask=mid + ask_offset,
                bid_offset=bid_offset,
                ask_offset=ask_offset,
                half_spread=half_spread,
                delta=float(row["delta"]),
                gamma=float(row["gamma"]),
                vega=float(row["vega"]),
                theta=float(row["theta"]),
                rho=float(row["rho"]),
            )
        )

    quotes = tuple(sorted(quotes, key=_quote_sort_key))
    summaries = _bucket_summaries(quotes, priced)
    return MultiQuoteResult(
        quotes=quotes,
        net_delta=float(net_delta),
        net_gamma=float(net_gamma),
        net_vega=float(net_vega),
        bucket_summary=summaries,
    )


def _quote_sort_key(quote: GreeksQuote) -> tuple:
    return (
        _BUCKET_ORDER.index(quote.bucket)
        if quote.bucket in _BUCKET_ORDER
        else len(_BUCKET_ORDER),
        quote.tau,
        quote.strike,
        quote.option_type,
        quote.key,
    )


def _bucket_summaries(
    quotes: tuple[GreeksQuote, ...],
    priced: list[dict],
) -> tuple[ExpiryBucketSummary, ...]:
    by_bucket: dict[str, dict] = {}
    for quote, row in zip(quotes, sorted(priced, key=lambda r: _quote_sort_key(_to_quote_for_sort(r)))):
        bucket = quote.bucket
        group = by_bucket.setdefault(
            bucket,
            {
                "n": 0,
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "width_total": 0.0,
            },
        )
        group["n"] += 1
        group["delta"] += row["delta_exposure"]
        group["gamma"] += row["gamma_exposure"]
        group["vega"] += row["vega_exposure"]
        group["width_total"] += quote.half_spread

    ordered_buckets = sorted(
        by_bucket,
        key=lambda b: (
            _BUCKET_ORDER.index(b)
            if b in _BUCKET_ORDER
            else len(_BUCKET_ORDER)
        ),
    )
    return tuple(
        ExpiryBucketSummary(
            bucket=bucket,
            n_contracts=group["n"],
            net_delta=float(group["delta"]),
            net_gamma=float(group["gamma"]),
            net_vega=float(group["vega"]),
            mean_half_spread=float(group["width_total"] / group["n"]),
        )
        for bucket in ordered_buckets
        for group in [by_bucket[bucket]]
    )


def _to_quote_for_sort(row: dict) -> GreeksQuote:
    """Build a lightweight quote solely for ordering priced rows."""
    contract = row["contract"]
    return GreeksQuote(
        key=contract.key,
        strike=contract.strike,
        tau=contract.tau,
        option_type=contract.option_type,
        bucket=expiry_bucket_id(contract.tau),
        mid=0.0,
        bid=0.0,
        ask=0.0,
        bid_offset=0.0,
        ask_offset=0.0,
        half_spread=0.0,
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        rho=0.0,
    )
