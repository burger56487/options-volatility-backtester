"""Account-engine-driven multi-contract market making with delta hedging.

This module wires the Greeks-aware quoting policies, the portfolio account,
the execution engine, daily delta hedging, financing and risk limits into one
event-style daily simulation.  A liquidity taker arrives on trading days with
configurable probability, buys from or sells to the posted book at the quoted
premium, and the book is then re-hedged towards delta neutrality.  Limits are
checked pre-trade for every option fill and post-trade each day; a daily-loss
or drawdown breach halts new quoting for the rest of the run.

This is a research-grade simulation of a synthetic option book, not live
trading: option mids are BSM fair values, taker behaviour is stylised and the
margin model is a simplified estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.domain.enums import OptionType
from src.execution.commission import CommissionSchedule
from src.execution.engine import ExecutionEngine
from src.execution.fill_model import ExecutionParameters
from src.execution.market_snapshot import MarketSnapshot as ExecutionSnapshot
from src.execution.metrics import fills_to_dataframe
from src.financing.margin import estimate_account_margin
from src.market_making.greeks_book import (
    BookContract,
    GreeksQuoteConfig,
    quote_multi_contract_book,
)
from src.portfolio.account import Account
from src.portfolio.fills import Fill
from src.portfolio.identifiers import InstrumentId, InstrumentType
from src.portfolio.orders import Order, Side
from src.portfolio.reconciliation import reconcile_pnl_bridge
from src.portfolio.valuation import (
    MarketSnapshot as PricingMarketSnapshot,
    create_account_snapshot,
)
from src.risk.exposures import calculate_portfolio_greeks
from src.risk.limits import (
    BLOCKING_ACTIONS,
    RiskLimits,
    check_portfolio_limits,
)
from src.risk.pre_trade import simulate_post_trade_limit_check


def default_market_making_limits(initial_capital: float) -> RiskLimits:
    """Sensible research-grade limits relative to the starting capital."""
    return RiskLimits(
        max_gross_exposure=20.0 * initial_capital,
        max_leverage=20.0,
        max_abs_delta=100_000.0,
        max_abs_gamma=10_000.0,
        max_abs_vega=1_000_000.0,
        max_daily_loss=0.02 * initial_capital,
        max_drawdown=0.10,
        min_cash_buffer=0.0,
    )


@dataclass(frozen=True)
class AccountMarketMakingConfig:
    symbol: str = "SPY"
    initial_capital: float = 200_000.0
    strikes: tuple[float, ...] = (0.95, 1.00, 1.05)
    expiry_days: tuple[int, ...] = (10, 20)
    include_puts: bool = True
    contracts_per_fill: float = 1.0
    multiplier: float = 100.0
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.012
    volatility: float = 0.25
    arrival_probability: float = 0.6
    arrival_side: str = "both"
    seed: int = 7
    quote_config: GreeksQuoteConfig = GreeksQuoteConfig()
    commission: CommissionSchedule = CommissionSchedule(
        per_share=0.005,
        per_contract=0.65,
        minimum=1.0,
    )
    execution: ExecutionParameters = ExecutionParameters(
        slippage_bps=1.0,
        impact_coefficient=0.0,
    )
    hedge_band_shares: float = 50.0
    annual_lending_rate: float = 0.03
    annual_borrowing_rate: float = 0.06
    annual_borrow_fee_rate: float = 0.01
    risk_limits: RiskLimits = field(
        default_factory=lambda: default_market_making_limits(200_000.0)
    )
    block_on_reduce: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.arrival_probability <= 1.0:
            raise ValueError("arrival_probability must lie in [0, 1].")
        if self.arrival_side not in {"both", "buy", "sell"}:
            raise ValueError("arrival_side must be 'both', 'buy' or 'sell'.")
        if not self.expiry_days or any(d < 1 for d in self.expiry_days):
            raise ValueError("expiry_days must be a non-empty tuple of days.")
        if not self.strikes or any(s <= 0 for s in self.strikes):
            raise ValueError("strikes must be positive.")


def _intrinsic(strike: float, option_type: OptionType, spot: float) -> float:
    if option_type == OptionType.CALL:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _option_instrument(symbol: str, expiry, strike: float, option_type: OptionType) -> InstrumentId:
    return InstrumentId(
        instrument_type=InstrumentType.OPTION,
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
    )


def _stock_instrument(symbol: str) -> InstrumentId:
    return InstrumentId(
        instrument_type=InstrumentType.STOCK,
        symbol=symbol,
    )


def run_account_market_making(
    price_data: pd.DataFrame,
    config: AccountMarketMakingConfig = AccountMarketMakingConfig(),
    entry_date=None,
) -> dict:
    """Run one daily-grid market-making episode through the account engine."""
    data = price_data.sort_index()
    dates = list(data.index)
    if entry_date is None:
        start_idx = 0
    else:
        if pd.Timestamp(entry_date) not in data.index:
            raise ValueError("entry_date must be in price_data.")
        start_idx = dates.index(pd.Timestamp(entry_date))
    expiry_indexes = tuple(
        min(start_idx + offset, len(data) - 1)
        for offset in config.expiry_days
    )
    expiry_indexes = tuple(sorted(set(expiry_indexes)))
    end_idx = max(expiry_indexes)
    if end_idx <= start_idx:
        raise ValueError("Insufficient price data for requested expiry days.")
    trade_index = data.index[start_idx : end_idx + 1]
    spot_entry = float(data["Close"].iloc[start_idx])

    legs = []
    for expiry_index in expiry_indexes:
        for strike_multiplier in config.strikes:
            strike = round(spot_entry * strike_multiplier, 2)
            for option_type in (
                [OptionType.CALL, OptionType.PUT]
                if config.include_puts
                else [OptionType.CALL]
            ):
                instrument = _option_instrument(
                    config.symbol,
                    trade_index[expiry_index - start_idx].date(),
                    strike,
                    option_type,
                )
                legs.append(
                    {
                        "key": instrument.key(),
                        "instrument": instrument,
                        "expiry_index": expiry_index,
                        "strike": strike,
                        "option_type": option_type,
                    }
                )
    legs = tuple(sorted(legs, key=lambda leg: (leg["expiry_index"], leg["strike"], leg["option_type"].value)))
    stock_id = _stock_instrument(config.symbol)
    stock_key = stock_id.key()
    leg_by_key = {leg["key"]: leg for leg in legs}

    account = Account(
        account_id="acct-mm",
        run_id="account-market-making",
        base_currency="USD",
        initial_capital=config.initial_capital,
    )
    engine = ExecutionEngine(
        commission_schedule=config.commission,
        parameters=config.execution,
        run_id="account-market-making",
    )
    rng = np.random.default_rng(config.seed)

    snapshots = []
    fills = []
    breach_log = []
    rejected_fill_count = 0
    halted = False
    halt_reasons = []
    previous_day_equity = config.initial_capital

    for position, timestamp in enumerate(trade_index):
        i = start_idx + position
        spot = float(data["Close"].iloc[i])
        date = timestamp.date()
        py_timestamp = timestamp.to_pydatetime()

        live_keys = [
            leg["key"]
            for leg in legs
            if i < leg["expiry_index"]
        ]
        contracts = []
        for leg in legs:
            if i >= leg["expiry_index"]:
                continue
            position_row = account.positions.get(leg["key"])
            quantity = float(position_row.quantity) if position_row else 0.0
            tau = (leg["expiry_index"] - i) / 252.0
            contracts.append(
                BookContract(
                    key=leg["key"],
                    strike=leg["strike"],
                    tau=max(tau, 1e-9),
                    option_type=leg["option_type"].value,
                    quantity=quantity,
                    multiplier=config.multiplier,
                )
            )
        quote_result = quote_multi_contract_book(
            spot=spot,
            contracts=contracts,
            config=config.quote_config,
        )
        quote_by_key = {quote.key: quote for quote in quote_result.quotes}

        mark_prices = {}
        for leg in legs:
            if i >= leg["expiry_index"]:
                mark_prices[leg["key"]] = _intrinsic(
                    leg["strike"],
                    leg["option_type"],
                    spot,
                )
            else:
                mark_prices[leg["key"]] = quote_by_key[leg["key"]].mid
        mark_prices[stock_key] = spot

        daily_pnl_so_far = (
            account.equity(mark_prices) - previous_day_equity
        )
        if not halted and rng.random() < config.arrival_probability and live_keys:
            key = str(rng.choice(live_keys))
            quote = quote_by_key[key]
            if config.arrival_side == "both":
                client_buy = bool(rng.random() < 0.5)
            else:
                client_buy = config.arrival_side == "buy"
            price = quote.ask if client_buy else quote.bid
            side = Side.SELL if client_buy else Side.BUY
            half_spread = quote.ask - quote.mid
            fill = Fill(
                fill_id=f"fill-mm-{len(fills) + 1}",
                order_id=f"order-mm-{position}",
                run_id="account-market-making",
                timestamp=py_timestamp,
                instrument_id=leg_by_key[key]["instrument"],
                side=side,
                quantity=config.contracts_per_fill,
                price=price,
                multiplier=config.multiplier,
                commission=config.commission.per_contract
                * config.contracts_per_fill,
                spread_cost=half_spread
                * config.contracts_per_fill
                * config.multiplier,
            )
            market = _pricing_snapshot(
                py_timestamp,
                quote_by_key,
                mark_prices,
                stock_key,
                config,
            )
            check = simulate_post_trade_limit_check(
                account=account,
                hypothetical_fill=fill,
                market=market,
                limits=config.risk_limits,
                daily_pnl=daily_pnl_so_far,
            )
            reduce_breach = any(
                breach.name in {"delta", "gamma", "vega"}
                for breach in check.breaches
            )
            if check.allowed and (not config.block_on_reduce or not reduce_breach):
                account.apply_fill(fill)
                fills.append(fill)
            else:
                rejected_fill_count += 1
                breach_log.append(
                    {
                        "timestamp": py_timestamp,
                        "event": "pre_trade_rejected",
                        "instrument": key,
                        "reasons": ", ".join(
                            breach.name for breach in check.breaches
                        )
                        or "limit_not_allowed",
                    }
                )

        market = _pricing_snapshot(
            py_timestamp,
            quote_by_key,
            mark_prices,
            stock_key,
            config,
        )
        portfolio_greeks = calculate_portfolio_greeks(account, market)
        stock_position = account.positions.get(stock_key)
        current_shares = float(
            stock_position.quantity if stock_position is not None else 0.0
        )
        target_shares = -portfolio_greeks.delta
        if (
            abs(target_shares - current_shares)
            > config.hedge_band_shares
        ):
            trade_shares = target_shares - current_shares
            order = Order(
                order_id=f"order-hedge-{position}",
                run_id="account-market-making",
                timestamp=py_timestamp,
                instrument_id=stock_id,
                side=Side.BUY if trade_shares > 0 else Side.SELL,
                quantity=abs(trade_shares),
            )
            execution = engine.execute(
                order,
                ExecutionSnapshot(
                    timestamp=py_timestamp,
                    symbol=config.symbol,
                    bid=spot,
                    ask=spot,
                    mid=spot,
                    orderbook_mode="mid_only",
                    quote_age_seconds=0.0,
                ),
                account=account,
            )
            fills.extend(execution.fills)

        account.accrue_financing(
            timestamp=py_timestamp,
            market_prices={stock_key: spot},
            annual_lending_rate=config.annual_lending_rate,
            annual_borrowing_rate=config.annual_borrowing_rate,
            annual_borrow_fee_rate=config.annual_borrow_fee_rate,
            year_fraction=1.0 / 365.0,
        )
        account.settle_expired_options(
            timestamp=py_timestamp,
            spot_prices={config.symbol: spot},
            as_of_date=date,
        )
        snapshot = create_account_snapshot(
            account,
            market,
            margin_estimate=estimate_account_margin(
                account,
                market_prices=mark_prices,
                underlying_spots={config.symbol: spot},
            ),
        )
        snapshots.append(snapshot)
        day_pnl = snapshot.equity - previous_day_equity
        previous_day_equity = snapshot.equity

        if not halted:
            day_check = check_portfolio_limits(
                gross_exposure=snapshot.gross_exposure,
                leverage=snapshot.leverage,
                delta=portfolio_greeks.delta,
                gamma=portfolio_greeks.gamma,
                vega=portfolio_greeks.vega,
                daily_pnl=day_pnl,
                drawdown=snapshot.drawdown,
                cash=snapshot.cash,
                limits=config.risk_limits,
            )
            blocking = [
                breach
                for breach in day_check.breaches
                if breach.action in BLOCKING_ACTIONS
            ]
            if blocking:
                halted = True
                halt_reasons = [breach.name for breach in blocking]
                for breach in blocking:
                    breach_log.append(
                        {
                            "timestamp": py_timestamp,
                            "event": "halt",
                            "instrument": "",
                            "reasons": breach.name,
                        }
                    )

    final_spot = float(data["Close"].iloc[end_idx])
    stock_position = account.positions.get(stock_key)
    if stock_position is not None and abs(stock_position.quantity) > 1e-9:
        side = Side.BUY if stock_position.quantity < 0 else Side.SELL
        engine.execute(
            Order(
                order_id="order-close-stock",
                run_id="account-market-making",
                timestamp=trade_index[-1].to_pydatetime(),
                instrument_id=stock_id,
                side=side,
                quantity=abs(stock_position.quantity),
            ),
            ExecutionSnapshot(
                timestamp=trade_index[-1].to_pydatetime(),
                symbol=config.symbol,
                bid=final_spot,
                ask=final_spot,
                mid=final_spot,
                orderbook_mode="mid_only",
                quote_age_seconds=0.0,
            ),
            account=account,
        )

    final_prices = {stock_key: final_spot}
    for leg in legs:
        key = leg["key"]
        if key in account.positions:
            if end_idx >= leg["expiry_index"]:
                final_prices[key] = _intrinsic(
                    leg["strike"],
                    leg["option_type"],
                    final_spot,
                )
            else:
                tau = max((leg["expiry_index"] - end_idx) / 252.0, 1e-9)
                quote = quote_multi_contract_book(
                    spot=final_spot,
                    contracts=[
                        BookContract(
                            key=key,
                            strike=leg["strike"],
                            tau=tau,
                            option_type=leg["option_type"].value,
                            quantity=0.0,
                            multiplier=config.multiplier,
                        )
                    ],
                    config=config.quote_config,
                ).quotes[0]
                final_prices[key] = quote.mid
    final_snapshot = create_account_snapshot(
        account,
        PricingMarketSnapshot(
            timestamp=trade_index[-1].to_pydatetime(),
            prices=final_prices,
            deltas={},
            gammas={},
            vegas={},
            thetas={},
            rhos={},
        ),
        margin_estimate=estimate_account_margin(
            account,
            market_prices=final_prices,
            underlying_spots={config.symbol: final_spot},
        ),
    )
    reconciliation = reconcile_pnl_bridge(account, final_prices)
    return {
        "final_equity": float(final_snapshot.equity),
        "total_pnl": float(
            final_snapshot.equity - account.initial_capital
        ),
        "reconciliation_passed": bool(reconciliation.passed),
        "reconciliation_difference": float(reconciliation.difference),
        "fill_count": len(fills),
        "rejected_fill_count": int(rejected_fill_count),
        "halted": bool(halted),
        "halt_reasons": halt_reasons,
        "snapshots": pd.DataFrame(
            [snapshot.to_dict() for snapshot in snapshots]
        ),
        "fills": fills_to_dataframe(fills),
        "breach_log": pd.DataFrame(breach_log),
    }
def _pricing_snapshot(
    timestamp: datetime,
    quote_by_key: dict,
    mark_prices: dict[str, float],
    stock_key: str,
    config: AccountMarketMakingConfig,
) -> PricingMarketSnapshot:
    deltas = {key: 0.0 for key in mark_prices}
    deltas[stock_key] = 1.0
    gammas = {key: 0.0 for key in mark_prices}
    vegas = {key: 0.0 for key in mark_prices}
    thetas = {key: 0.0 for key in mark_prices}
    rhos = {key: 0.0 for key in mark_prices}
    for key, quote in quote_by_key.items():
        deltas[key] = quote.delta * config.multiplier
        gammas[key] = quote.gamma * config.multiplier
        vegas[key] = quote.vega * config.multiplier
        thetas[key] = quote.theta * config.multiplier
        rhos[key] = quote.rho * config.multiplier
    return PricingMarketSnapshot(
        timestamp=timestamp,
        prices=dict(mark_prices),
        deltas=deltas,
        gammas=gammas,
        vegas=vegas,
        thetas=thetas,
        rhos=rhos,
    )


def save_account_market_making_result(
    result: dict,
    output_directory: str | Path,
) -> Path:
    """Write snapshots, fills and summary for an account MM run."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    result["snapshots"].to_csv(
        output_path / "snapshots.csv",
        index=False,
    )
    result["fills"].to_csv(output_path / "fills.csv", index=False)
    if not result["breach_log"].empty:
        result["breach_log"].to_csv(
            output_path / "breach_log.csv",
            index=False,
        )
    import json

    summary = {
        "final_equity": result["final_equity"],
        "total_pnl": result["total_pnl"],
        "reconciliation_passed": result["reconciliation_passed"],
        "reconciliation_difference": result[
            "reconciliation_difference"
        ],
        "fill_count": result["fill_count"],
        "rejected_fill_count": result["rejected_fill_count"],
        "halted": result["halted"],
        "halt_reasons": result["halt_reasons"],
    }
    with (output_path / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return output_path
