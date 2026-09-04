"""Account-engine-driven delta hedge: integration milestone.

This module wires the portfolio Account, execution engine and risk limits
into a daily mark-to-market loop for a single long call plus an underlying
delta hedge. It is the first integration slice on the path to replacing the
legacy close-to-close backtest; option Greeks come from the analytic BSM
engine (per-share Greeks scaled to per-contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.domain.enums import OptionType
from src.execution.commission import CommissionSchedule
from src.execution.engine import ExecutionEngine
from src.execution.fill_model import ExecutionParameters
from src.execution.market_snapshot import MarketSnapshot
from src.execution.metrics import fills_to_dataframe
from src.portfolio.account import Account
from src.portfolio.fills import Fill
from src.portfolio.identifiers import (
    InstrumentId,
    InstrumentType,
)
from src.portfolio.orders import Order, Side
from src.portfolio.reconciliation import reconcile_pnl_bridge
from src.portfolio.reconciliation import ledger_type_total
from src.portfolio.cash_ledger import CashFlowType
from src.portfolio.valuation import (
    AccountSnapshot,
    MarketSnapshot as PricingMarketSnapshot,
    create_account_snapshot,
)
from src.pricing.black_scholes import price_and_greeks


@dataclass(frozen=True)
class AccountHedgeConfig:
    symbol: str = "SPY"
    initial_capital: float = 100_000.0
    option_quantity: int = 1
    multiplier: int = 100
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.012
    volatility: float = 0.25
    expiry_days: int = 30
    strike_multiplier: float = 1.0
    commission: CommissionSchedule = CommissionSchedule(
        per_share=0.005,
        per_contract=0.65,
        minimum=1.0,
    )
    execution: ExecutionParameters = ExecutionParameters(
        slippage_bps=1.0,
        impact_coefficient=0.0,
    )
    annual_lending_rate: float = 0.03
    annual_borrowing_rate: float = 0.06
    annual_borrow_fee_rate: float = 0.01


def _contract_id(symbol: str, expiry, strike: float):
    return InstrumentId(
        instrument_type=InstrumentType.OPTION,
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        option_type=OptionType.CALL,
    )


def _option_id_for_type(symbol: str, expiry, strike: float, option_type):
    return InstrumentId(
        instrument_type=InstrumentType.OPTION,
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
    )


def _stock_id(symbol: str) -> InstrumentId:
    return InstrumentId(
        instrument_type=InstrumentType.STOCK,
        symbol=symbol,
    )


def run_account_delta_hedge(
    price_data: pd.DataFrame,
    entry_date,
    config: AccountHedgeConfig = AccountHedgeConfig(),
) -> dict:
    """Run one account-engine-driven call-plus-hedge trade."""
    data = price_data.sort_index()
    dates = list(data.index)
    if pd.Timestamp(entry_date) not in data.index:
        raise ValueError("entry_date must be in price_data.")
    start_idx = list(data.index).index(pd.Timestamp(entry_date))
    end_idx = min(
        len(data) - 1,
        start_idx + config.expiry_days,
    )
    trade_index = data.index[start_idx : end_idx + 1]
    spot_entry = float(data["Close"].iloc[start_idx])
    strike = round(spot_entry * config.strike_multiplier, 2)
    expiry = trade_index[-1]
    option_id = _contract_id(config.symbol, expiry, strike)
    stock_id = _stock_id(config.symbol)

    account = Account(
        account_id="acct-1",
        run_id="account-hedge",
        base_currency="USD",
        initial_capital=config.initial_capital,
    )
    engine = ExecutionEngine(
        commission_schedule=config.commission,
        parameters=config.execution,
        run_id="account-hedge",
    )

    def premium(spot: float, time_to_expiry: float) -> float:
        result = price_and_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=config.risk_free_rate,
            volatility=config.volatility,
            option_type="call",
            dividend_yield=config.dividend_yield,
        )
        return result

    snapshots = []
    fills = []
    entry_done = False
    for position, (timestamp, row) in enumerate(
        zip(trade_index, data.loc[trade_index].itertuples())
    ):
        spot = float(row.Close)
        remaining = max(
            (len(trade_index) - 1 - position) / 252.0,
            1e-6,
        )
        greeks = premium(spot, remaining)
        if not entry_done:
            ask = greeks.price
            option_fill = Fill(
                fill_id=f"fill-entry-{position}",
                order_id="order-entry",
                run_id="account-hedge",
                timestamp=timestamp.to_pydatetime(),
                instrument_id=option_id,
                side=Side.BUY,
                quantity=float(config.option_quantity),
                price=ask,
                multiplier=float(config.multiplier),
                commission=config.commission.per_contract
                * config.option_quantity,
            )
            account.apply_fill(option_fill)
            fills.append(option_fill)
            entry_done = True

        option_delta = greeks.delta * config.option_quantity * config.multiplier
        stock_position = account.positions.get(stock_id.key())
        current_shares = float(
            stock_position.quantity if stock_position is not None else 0.0
        )
        target_shares = -option_delta
        trade_shares = target_shares - current_shares
        if abs(trade_shares) > 1e-9:
            side = Side.BUY if trade_shares > 0 else Side.SELL
            order = Order(
                order_id=f"order-hedge-{position}",
                run_id="account-hedge",
                timestamp=timestamp.to_pydatetime(),
                instrument_id=stock_id,
                side=side,
                quantity=abs(trade_shares),
            )
            snapshot = MarketSnapshot(
                timestamp=timestamp.to_pydatetime(),
                symbol=config.symbol,
                bid=spot,
                ask=spot,
                mid=spot,
                orderbook_mode="mid_only",
                quote_age_seconds=0.0,
            )
            execution = engine.execute(order, snapshot, account=account)
            fills.extend(execution.fills)

        option_price = premium(spot, remaining).price
        account.accrue_financing(
            timestamp=timestamp.to_pydatetime(),
            market_prices={stock_id.key(): spot},
            annual_lending_rate=config.annual_lending_rate,
            annual_borrowing_rate=config.annual_borrowing_rate,
            annual_borrow_fee_rate=config.annual_borrow_fee_rate,
            year_fraction=1.0 / 365.0,
        )
        market = PricingMarketSnapshot(
            timestamp=timestamp.to_pydatetime(),
            prices={option_id.key(): option_price, stock_id.key(): spot},
            deltas={
                option_id.key(): greeks.delta
                * config.multiplier,
                stock_id.key(): 1.0,
            },
            gammas={
                option_id.key(): greeks.gamma * config.multiplier,
                stock_id.key(): 0.0,
            },
            vegas={
                option_id.key(): greeks.vega * config.multiplier,
                stock_id.key(): 0.0,
            },
            thetas={
                option_id.key(): greeks.theta * config.multiplier,
                stock_id.key(): 0.0,
            },
            rhos={
                option_id.key(): greeks.rho * config.multiplier,
                stock_id.key(): 0.0,
            },
        )
        snapshots.append(create_account_snapshot(account, market))

    # Expiry settlement and stock close-out at the final mark.
    final_spot = float(data["Close"].iloc[end_idx])
    account.settle_expired_options(
        timestamp=trade_index[-1].to_pydatetime(),
        spot_prices={config.symbol: final_spot},
        as_of_date=trade_index[-1].date(),
    )
    stock_position = account.positions.get(stock_id.key())
    if stock_position is not None and abs(stock_position.quantity) > 1e-9:
        side = Side.BUY if stock_position.quantity < 0 else Side.SELL
        close_order = Order(
            order_id="order-close-stock",
            run_id="account-hedge",
            timestamp=trade_index[-1].to_pydatetime(),
            instrument_id=stock_id,
            side=side,
            quantity=abs(stock_position.quantity),
        )
        execution = engine.execute(
            close_order,
            MarketSnapshot(
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
        fills.extend(execution.fills)
    final_prices = {
        option_id.key(): max(final_spot - strike, 0.0),
        stock_id.key(): final_spot,
    }
    final_snapshot = create_account_snapshot(
        account,
        PricingMarketSnapshot(
            timestamp=trade_index[-1].to_pydatetime(),
            prices=final_prices,
            deltas={}, gammas={}, vegas={}, thetas={}, rhos={},
        ),
    )
    reconciliation = reconcile_pnl_bridge(
        account,
        market_prices=final_prices,
    )
    return {
        "final_equity": float(final_snapshot.equity),
        "total_pnl": float(final_snapshot.equity - account.initial_capital),
        "reconciliation_passed": bool(reconciliation.passed),
        "reconciliation_difference": float(reconciliation.difference),
        "bridge_debug": {
            "cash": float(account.cash),
            "realised_pnl": float(
                sum(p.realised_pnl for p in account.positions.values())
            ),
            "unrealised_pnl": float(
                sum(
                    p.unrealised_pnl(final_prices[key])
                    for key, p in account.positions.items()
                    if key in final_prices
                )
            ),
            "interest": float(
                ledger_type_total(account, CashFlowType.INTEREST)
            ),
            "financing": float(
                ledger_type_total(account, CashFlowType.FINANCING)
            ),
            "borrow_fees": float(
                ledger_type_total(account, CashFlowType.BORROW_FEE)
            ),
            "settlement": float(
                ledger_type_total(
                    account, CashFlowType.OPTION_SETTLEMENT
                )
            ),
            "fees_paid": float(account.fees_paid),
        },
        "snapshots": pd.DataFrame(
            [snapshot.to_dict() for snapshot in snapshots]
        ),
        "fills": fills_to_dataframe(fills),
    }


def run_account_straddle_backtest(
    price_data: pd.DataFrame,
    entry_date,
    config: AccountHedgeConfig = AccountHedgeConfig(),
) -> dict:
    """Run one account-engine-driven ATM call+put straddle with delta hedge."""
    data = price_data.sort_index()
    dates = list(data.index)
    if pd.Timestamp(entry_date) not in data.index:
        raise ValueError("entry_date must be in price_data.")
    start_idx = dates.index(pd.Timestamp(entry_date))
    end_idx = min(len(data) - 1, start_idx + config.expiry_days)
    trade_index = data.index[start_idx : end_idx + 1]
    spot_entry = float(data["Close"].iloc[start_idx])
    strike = round(spot_entry * config.strike_multiplier, 2)
    expiry = trade_index[-1]
    call_id = _option_id_for_type(
        config.symbol, expiry, strike, OptionType.CALL
    )
    put_id = _option_id_for_type(
        config.symbol, expiry, strike, OptionType.PUT
    )
    stock_id = _stock_id(config.symbol)

    account = Account(
        account_id="acct-straddle",
        run_id="account-straddle",
        base_currency="USD",
        initial_capital=config.initial_capital,
    )
    engine = ExecutionEngine(
        commission_schedule=config.commission,
        parameters=config.execution,
        run_id="account-straddle",
    )

    def leg(option_type, spot: float, tau: float):
        return price_and_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=tau,
            risk_free_rate=config.risk_free_rate,
            volatility=config.volatility,
            option_type=option_type.value,
            dividend_yield=config.dividend_yield,
        )

    snapshots = []
    fills = []
    entered = False
    for position, timestamp in enumerate(trade_index):
        spot = float(data.loc[timestamp, "Close"])
        remaining = max(
            (len(trade_index) - 1 - position) / 252.0,
            1e-6,
        )
        call_greeks = leg(OptionType.CALL, spot, remaining)
        put_greeks = leg(OptionType.PUT, spot, remaining)

        if not entered:
            for option_id, greeks in [
                (call_id, call_greeks),
                (put_id, put_greeks),
            ]:
                option_fill = Fill(
                    fill_id=f"fill-entry-{option_id.option_type.value}",
                    order_id="order-entry",
                    run_id="account-straddle",
                    timestamp=timestamp.to_pydatetime(),
                    instrument_id=option_id,
                    side=Side.BUY,
                    quantity=float(config.option_quantity),
                    price=greeks.price,
                    multiplier=float(config.multiplier),
                    commission=config.commission.per_contract
                    * config.option_quantity,
                )
                account.apply_fill(option_fill)
                fills.append(option_fill)
            entered = True

        combined_delta = (
            (call_greeks.delta + put_greeks.delta)
            * config.option_quantity
            * config.multiplier
        )
        stock_position = account.positions.get(stock_id.key())
        current_shares = float(
            stock_position.quantity if stock_position is not None else 0.0
        )
        trade_shares = -combined_delta - current_shares
        if abs(trade_shares) > 1e-9:
            side = Side.BUY if trade_shares > 0 else Side.SELL
            order = Order(
                order_id=f"order-hedge-{position}",
                run_id="account-straddle",
                timestamp=timestamp.to_pydatetime(),
                instrument_id=stock_id,
                side=side,
                quantity=abs(trade_shares),
            )
            execution = engine.execute(
                order,
                MarketSnapshot(
                    timestamp=timestamp.to_pydatetime(),
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

        prices = {
            call_id.key(): leg(OptionType.CALL, spot, remaining).price,
            put_id.key(): leg(OptionType.PUT, spot, remaining).price,
            stock_id.key(): spot,
        }
        account.accrue_financing(
            timestamp=timestamp.to_pydatetime(),
            market_prices={stock_id.key(): spot},
            annual_lending_rate=config.annual_lending_rate,
            annual_borrowing_rate=config.annual_borrowing_rate,
            annual_borrow_fee_rate=config.annual_borrow_fee_rate,
            year_fraction=1.0 / 365.0,
        )

        deltas = {
            call_id.key(): call_greeks.delta * config.multiplier,
            put_id.key(): put_greeks.delta * config.multiplier,
            stock_id.key(): 1.0,
        }
        gammas = {
            call_id.key(): call_greeks.gamma * config.multiplier,
            put_id.key(): put_greeks.gamma * config.multiplier,
            stock_id.key(): 0.0,
        }
        vegas = {
            call_id.key(): call_greeks.vega * config.multiplier,
            put_id.key(): put_greeks.vega * config.multiplier,
            stock_id.key(): 0.0,
        }
        thetas = {
            call_id.key(): call_greeks.theta * config.multiplier,
            put_id.key(): put_greeks.theta * config.multiplier,
            stock_id.key(): 0.0,
        }
        rhos = {
            call_id.key(): call_greeks.rho * config.multiplier,
            put_id.key(): put_greeks.rho * config.multiplier,
            stock_id.key(): 0.0,
        }
        snapshots.append(
            create_account_snapshot(
                account,
                PricingMarketSnapshot(
                    timestamp=timestamp.to_pydatetime(),
                    prices=prices,
                    deltas=deltas,
                    gammas=gammas,
                    vegas=vegas,
                    thetas=thetas,
                    rhos=rhos,
                ),
            )
        )

    final_spot = float(data["Close"].iloc[end_idx])
    account.settle_expired_options(
        timestamp=trade_index[-1].to_pydatetime(),
        spot_prices={config.symbol: final_spot},
        as_of_date=trade_index[-1].date(),
    )
    stock_position = account.positions.get(stock_id.key())
    if stock_position is not None and abs(stock_position.quantity) > 1e-9:
        side = Side.BUY if stock_position.quantity < 0 else Side.SELL
        execution = engine.execute(
            Order(
                order_id="order-close-stock",
                run_id="account-straddle",
                timestamp=trade_index[-1].to_pydatetime(),
                instrument_id=stock_id,
                side=side,
                quantity=abs(stock_position.quantity),
            ),
            MarketSnapshot(
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
        fills.extend(execution.fills)

    final_prices = {
        call_id.key(): max(final_spot - strike, 0.0),
        put_id.key(): max(strike - final_spot, 0.0),
        stock_id.key(): final_spot,
    }
    reconciliation = reconcile_pnl_bridge(account, final_prices)
    final_equity = account.equity(final_prices)
    return {
        "final_equity": float(final_equity),
        "total_pnl": float(final_equity - account.initial_capital),
        "reconciliation_passed": bool(reconciliation.passed),
        "reconciliation_difference": float(reconciliation.difference),
        "snapshots": pd.DataFrame(
            [snapshot.to_dict() for snapshot in snapshots]
        ),
        "fills": fills_to_dataframe(fills),
    }


def save_account_hedge_result(
    result: dict,
    output_directory: str | Path,
) -> Path:
    """Write snapshots, fills and a small summary for an account hedge run."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    result["snapshots"].to_csv(
        output_path / "account_snapshots.csv",
        index=False,
    )
    result["fills"].to_csv(
        output_path / "fills.csv",
        index=False,
    )
    summary = {
        "final_equity": result["final_equity"],
        "total_pnl": result["total_pnl"],
        "reconciliation_passed": result["reconciliation_passed"],
        "reconciliation_difference": result[
            "reconciliation_difference"
        ],
    }
    import json

    with (output_path / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return output_path
