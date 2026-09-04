"""Cash- and position-level account."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from .cash_ledger import (
    CashFlowType,
    CashLedgerEntry,
    calculate_cash_balance,
)
from .fills import Fill
from .identifiers import InstrumentId, InstrumentType
from .positions import Position


@dataclass
class Account:
    account_id: str
    run_id: str
    base_currency: str
    initial_capital: float
    positions: dict[str, Position] = field(default_factory=dict)
    cash_ledger: list[CashLedgerEntry] = field(default_factory=list)
    fees_paid: float = 0.0
    financing_paid: float = 0.0
    borrow_fees_paid: float = 0.0
    peak_equity: float | None = None

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        self.cash_ledger.append(
            CashLedgerEntry(
                entry_id=str(uuid4()),
                run_id=self.run_id,
                timestamp=datetime.min,
                currency=self.base_currency,
                amount=self.initial_capital,
                cash_flow_type=CashFlowType.INITIAL_CAPITAL,
                description="initial capital",
            )
        )

    @property
    def cash(self) -> float:
        return calculate_cash_balance(
            self.cash_ledger,
            self.base_currency,
        )

    def get_or_create_position(
        self,
        instrument_id: InstrumentId,
        multiplier: float,
    ) -> Position:
        key = instrument_id.key()
        if key not in self.positions:
            self.positions[key] = Position(
                instrument_id=instrument_id,
                multiplier=multiplier,
            )
        return self.positions[key]

    def apply_fill(self, fill: Fill) -> None:
        position = self.get_or_create_position(
            instrument_id=fill.instrument_id,
            multiplier=fill.multiplier,
        )
        position.apply_fill(fill)
        self.cash_ledger.append(
            CashLedgerEntry(
                entry_id=str(uuid4()),
                run_id=self.run_id,
                timestamp=fill.timestamp,
                currency=self.base_currency,
                amount=fill.cash_flow,
                cash_flow_type=CashFlowType.TRADE,
                description=f"fill {fill.instrument_id.key()}",
                related_order_id=fill.order_id,
                related_fill_id=fill.fill_id,
            )
        )
        self.fees_paid += fill.commission

    def _append_cash(
        self,
        timestamp: datetime,
        amount: float,
        cash_flow_type: CashFlowType,
        description: str,
    ) -> None:
        self.cash_ledger.append(
            CashLedgerEntry(
                entry_id=str(uuid4()),
                run_id=self.run_id,
                timestamp=timestamp,
                currency=self.base_currency,
                amount=amount,
                cash_flow_type=cash_flow_type,
                description=description,
            )
        )

    def market_value(self, market_prices: dict[str, float]) -> float:
        total = 0.0
        for key, position in self.positions.items():
            if key not in market_prices:
                raise KeyError(f"Missing market price for {key}")
            total += position.market_value(market_prices[key])
        return total

    def equity(self, market_prices: dict[str, float]) -> float:
        return self.cash + self.market_value(market_prices)

    def gross_exposure(self, market_prices: dict[str, float]) -> float:
        return sum(
            abs(position.market_value(market_prices[key]))
            for key, position in self.positions.items()
        )

    def net_exposure(self, market_prices: dict[str, float]) -> float:
        return self.market_value(market_prices)

    def leverage(self, market_prices: dict[str, float]) -> float:
        equity = self.equity(market_prices)
        if equity <= 0:
            return float("inf")
        return self.gross_exposure(market_prices) / equity

    def drawdown(self, market_prices: dict[str, float]) -> float:
        current_equity = self.equity(market_prices)
        if self.peak_equity is None:
            self.peak_equity = current_equity
        self.peak_equity = max(self.peak_equity, current_equity)
        if self.peak_equity <= 0:
            return 0.0
        return current_equity / self.peak_equity - 1.0

    def accrue_financing(
        self,
        timestamp: datetime,
        market_prices: dict[str, float],
        annual_lending_rate: float,
        annual_borrowing_rate: float,
        annual_borrow_fee_rate: float,
        year_fraction: float,
    ) -> None:
        """Accrue one period of cash interest and stock borrow fees."""
        cash = self.cash
        if cash >= 0:
            interest = cash * annual_lending_rate * year_fraction
            if interest:
                self._append_cash(
                    timestamp,
                    interest,
                    CashFlowType.INTEREST,
                    "cash lending interest",
                )
        else:
            financing_cost = (
                cash * annual_borrowing_rate * year_fraction
            )
            self._append_cash(
                timestamp,
                financing_cost,
                CashFlowType.FINANCING,
                "cash borrowing cost",
            )
            self.financing_paid += abs(financing_cost)

        for key, position in self.positions.items():
            if (
                position.quantity < 0
                and position.instrument_id.instrument_type
                == InstrumentType.STOCK
            ):
                short_market_value = position.market_value(
                    market_prices[key]
                )
                fee = (
                    abs(short_market_value)
                    * annual_borrow_fee_rate
                    * year_fraction
                )
                if fee:
                    self._append_cash(
                        timestamp,
                        -fee,
                        CashFlowType.BORROW_FEE,
                        "stock borrow fee",
                    )
                    self.borrow_fees_paid += fee

    def settle_expired_options(
        self,
        timestamp: datetime,
        spot_prices: dict[str, float],
        as_of_date,
    ) -> None:
        """Cash-settle expired option positions and zero them."""
        from datetime import date

        for key, position in list(self.positions.items()):
            instrument = position.instrument_id
            if (
                instrument.instrument_type != InstrumentType.OPTION
                or instrument.expiry is None
                or instrument.expiry > as_of_date
                or instrument.strike is None
            ):
                continue
            spot = spot_prices[instrument.symbol]
            if instrument.option_type.value == "call":
                intrinsic = max(spot - instrument.strike, 0.0)
            else:
                intrinsic = max(instrument.strike - spot, 0.0)
            settlement = (
                position.quantity
                * intrinsic
                * position.multiplier
            )
            if settlement:
                self._append_cash(
                    timestamp,
                    settlement,
                    CashFlowType.OPTION_SETTLEMENT,
                    f"settlement {instrument.key()}",
                )
            self.positions[key].quantity = 0.0
            self.positions[key].average_cost = 0.0
