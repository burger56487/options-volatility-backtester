"""Simple financing-rate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FinancingResult:
    start_time: datetime
    end_time: datetime
    cash_balance: float
    annual_rate: float
    year_fraction: float
    interest: float


def accrue_cash_interest(
    cash_balance: float,
    annual_lending_rate: float,
    annual_borrowing_rate: float,
    year_fraction: float,
) -> float:
    if year_fraction < 0:
        raise ValueError("Year fraction cannot be negative.")
    rate = (
        annual_lending_rate
        if cash_balance >= 0
        else annual_borrowing_rate
    )
    return cash_balance * rate * year_fraction


def calculate_stock_borrow_fee(
    short_market_value: float,
    annual_borrow_fee_rate: float,
    year_fraction: float,
) -> float:
    if short_market_value >= 0:
        return 0.0
    return (
        abs(short_market_value)
        * annual_borrow_fee_rate
        * year_fraction
    )
