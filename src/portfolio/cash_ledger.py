"""Append-only cash ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class CashFlowType(str, Enum):
    INITIAL_CAPITAL = "initial_capital"
    TRADE = "trade"
    COMMISSION = "commission"
    INTEREST = "interest"
    FINANCING = "financing"
    BORROW_FEE = "borrow_fee"
    DIVIDEND = "dividend"
    OPTION_SETTLEMENT = "option_settlement"
    MARGIN = "margin"
    ADJUSTMENT = "adjustment"


@dataclass(frozen=True)
class CashLedgerEntry:
    entry_id: str
    run_id: str
    timestamp: datetime
    currency: str
    amount: float
    cash_flow_type: CashFlowType
    description: str
    related_order_id: str | None = None
    related_fill_id: str | None = None


def calculate_cash_balance(
    entries: list[CashLedgerEntry],
    currency: str = "USD",
) -> float:
    return sum(
        entry.amount
        for entry in entries
        if entry.currency == currency
    )
