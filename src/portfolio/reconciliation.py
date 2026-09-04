"""Independent PnL-bridge reconciliation."""

from __future__ import annotations

from dataclasses import dataclass

from .account import Account
from .cash_ledger import CashFlowType


@dataclass(frozen=True)
class ReconciliationResult:
    passed: bool
    equity: float
    expected_equity: float
    difference: float
    tolerance: float


def ledger_type_total(
    account: Account,
    cash_flow_type: CashFlowType,
) -> float:
    return sum(
        entry.amount
        for entry in account.cash_ledger
        if entry.cash_flow_type == cash_flow_type
        and entry.currency == account.base_currency
    )


def reconcile_pnl_bridge(
    account: Account,
    market_prices: dict[str, float],
    tolerance: float = 1e-6,
) -> ReconciliationResult:
    """Check equity change against an independently built PnL bridge."""
    equity = account.equity(market_prices)
    total_pnl = equity - account.initial_capital

    realised = sum(
        position.realised_pnl
        for position in account.positions.values()
    )
    unrealised = sum(
        position.unrealised_pnl(market_prices[key])
        for key, position in account.positions.items()
    )
    financing = ledger_type_total(
        account, CashFlowType.FINANCING
    ) + ledger_type_total(account, CashFlowType.INTEREST)
    borrow_fees = ledger_type_total(
        account, CashFlowType.BORROW_FEE
    )
    bridge_pnl = (
        realised
        + unrealised
        + financing
        + borrow_fees
        - account.fees_paid
    )
    difference = total_pnl - bridge_pnl
    return ReconciliationResult(
        passed=abs(difference) <= tolerance,
        equity=equity,
        expected_equity=account.initial_capital + bridge_pnl,
        difference=difference,
        tolerance=tolerance,
    )
