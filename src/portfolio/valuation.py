"""Market and account snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from .account import Account


@dataclass(frozen=True)
class MarketSnapshot:
    timestamp: datetime
    prices: dict[str, float]
    # Greeks per contract (option delta/gamma/vega/theta/rho already include
    # the contract multiplier; stock delta is 1.0 per share).
    deltas: dict[str, float]
    gammas: dict[str, float]
    vegas: dict[str, float]
    thetas: dict[str, float]
    rhos: dict[str, float]


@dataclass(frozen=True)
class AccountSnapshot:
    timestamp: datetime
    cash: float
    market_value: float
    equity: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    drawdown: float
    realised_pnl: float
    unrealised_pnl: float
    fees_paid: float
    financing_paid: float
    borrow_fees_paid: float
    margin_used: float = 0.0
    available_capital: float = 0.0

    def to_dict(self) -> dict:
        output = asdict(self)
        output["timestamp"] = self.timestamp.isoformat()
        return output


def create_account_snapshot(
    account: Account,
    market: MarketSnapshot,
    margin_estimate: dict | None = None,
) -> AccountSnapshot:
    realised_pnl = sum(
        position.realised_pnl
        for position in account.positions.values()
    )
    unrealised_pnl = sum(
        position.unrealised_pnl(market.prices[key])
        for key, position in account.positions.items()
    )
    equity = account.equity(market.prices)
    margin_used = float(
        (margin_estimate or {}).get("initial_margin_total", 0.0)
    )
    return AccountSnapshot(
        timestamp=market.timestamp,
        cash=account.cash,
        market_value=account.market_value(market.prices),
        equity=equity,
        gross_exposure=account.gross_exposure(market.prices),
        net_exposure=account.net_exposure(market.prices),
        leverage=account.leverage(market.prices),
        drawdown=account.drawdown(market.prices),
        realised_pnl=realised_pnl,
        unrealised_pnl=unrealised_pnl,
        fees_paid=account.fees_paid,
        financing_paid=account.financing_paid,
        borrow_fees_paid=account.borrow_fees_paid,
        margin_used=margin_used,
        available_capital=equity - margin_used,
    )
