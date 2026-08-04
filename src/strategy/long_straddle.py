from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Literal

import pandas as pd

from src.pricing.black_scholes import (
    BlackScholesResult,
    price_and_greeks,
)


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionContract:
    """Definition of a European option contract."""

    option_type: OptionType
    strike: float
    expiry_date: pd.Timestamp
    multiplier: int = 100

    def __post_init__(self) -> None:
        if self.option_type not in {"call", "put"}:
            raise ValueError(
                "option_type must be either 'call' or 'put'."
            )

        if self.strike <= 0:
            raise ValueError("strike must be positive.")

        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1.")

        if pd.isna(self.expiry_date):
            raise ValueError("expiry_date must not be missing.")


@dataclass(frozen=True)
class OptionPosition:
    """
    An option position with entry price and quantity.

    Positive quantity represents a long option position.
    Negative quantity represents a short option position.
    """

    contract: OptionContract
    quantity: int
    entry_price: float

    def __post_init__(self) -> None:
        if self.quantity == 0:
            raise ValueError("quantity must not be zero.")

        if self.entry_price < 0:
            raise ValueError("entry_price must be non-negative.")

    @property
    def entry_notional(self) -> float:
        """Signed entry cash flow before fees."""
        return (
            -self.quantity
            * self.entry_price
            * self.contract.multiplier
        )

    def days_to_expiry(
        self,
        valuation_date: pd.Timestamp,
    ) -> int:
        """Calculate non-negative calendar days remaining to expiry."""
        if pd.isna(valuation_date):
            raise ValueError(
                "valuation_date must not be missing."
            )

        return max(
            0,
            int(
                (
                    self.contract.expiry_date.normalize()
                    - valuation_date.normalize()
                ).days
            ),
        )

    def time_to_expiry(
        self,
        valuation_date: pd.Timestamp,
    ) -> float:
        """Calculate remaining time to expiry in year fractions."""
        return self.days_to_expiry(valuation_date) / 365.0

    def intrinsic_value(
        self,
        spot: float,
    ) -> float:
        """Calculate per-unit option intrinsic value."""
        if spot <= 0:
            raise ValueError("spot must be positive.")

        if self.contract.option_type == "call":
            return max(spot - self.contract.strike, 0.0)

        return max(self.contract.strike - spot, 0.0)

    def price_and_greeks(
        self,
        valuation_date: pd.Timestamp,
        spot: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
    ) -> BlackScholesResult:
        """
        Value the position's contract at a given date.

        At expiry, returns intrinsic value and zero Greeks.
        """
        if spot <= 0:
            raise ValueError("spot must be positive.")

        if valuation_date > self.contract.expiry_date:
            intrinsic = self.intrinsic_value(spot)

            return BlackScholesResult(
                price=intrinsic,
                delta=0.0,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=0.0,
            )

        remaining_time = self.time_to_expiry(valuation_date)

        if remaining_time == 0:
            intrinsic = self.intrinsic_value(spot)

            return BlackScholesResult(
                price=intrinsic,
                delta=0.0,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=0.0,
            )

        return price_and_greeks(
            spot=spot,
            strike=self.contract.strike,
            time_to_expiry=remaining_time,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            option_type=self.contract.option_type,
            dividend_yield=dividend_yield,
        )

    def market_value(
        self,
        valuation_date: pd.Timestamp,
        spot: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Return signed mark-to-market value of the option position."""
        result = self.price_and_greeks(
            valuation_date=valuation_date,
            spot=spot,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
        )

        return (
            self.quantity
            * result.price
            * self.contract.multiplier
        )

    def pnl(
        self,
        valuation_date: pd.Timestamp,
        spot: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Return option mark-to-market P&L before transaction fees."""
        return self.market_value(
            valuation_date=valuation_date,
            spot=spot,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            dividend_yield=dividend_yield,
        ) + self.entry_notional


@dataclass(frozen=True)
class LongStraddle:
    """Long ATM call and put with the same strike, expiry, and quantity."""

    call_position: OptionPosition
    put_position: OptionPosition

    def __post_init__(self) -> None:
        call = self.call_position
        put = self.put_position

        if call.contract.option_type != "call":
            raise ValueError(
                "call_position must contain a call option."
            )

        if put.contract.option_type != "put":
            raise ValueError(
                "put_position must contain a put option."
            )

        if not isclose(
            call.contract.strike,
            put.contract.strike,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "call and put strikes must match."
            )

        if call.contract.expiry_date != put.contract.expiry_date:
            raise ValueError(
                "call and put expiry dates must match."
            )

        if call.contract.multiplier != put.contract.multiplier:
            raise ValueError(
                "call and put multipliers must match."
            )

        if call.quantity != put.quantity:
            raise ValueError(
                "call and put quantities must match."
            )

        if call.quantity < 1:
            raise ValueError(
                "LongStraddle requires positive option quantities."
            )

    @property
    def quantity(self) -> int:
        """Number of call-put straddles."""
        return self.call_position.quantity

    @property
    def strike(self) -> float:
        """Shared straddle strike."""
        return self.call_position.contract.strike

    @property
    def expiry_date(self) -> pd.Timestamp:
        """Shared straddle expiry."""
        return self.call_position.contract.expiry_date

    @property
    def entry_cost(self) -> float:
        """Positive cash cost to establish the long straddle."""
        return -(
            self.call_position.entry_notional
            + self.put_position.entry_notional
        )

    def market_value(
        self,
        valuation_date: pd.Timestamp,
        spot: float,
        risk_free_rate: float,
        call_volatility: float,
        put_volatility: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Calculate total mark-to-market value of call and put positions."""
        call_value = self.call_position.market_value(
            valuation_date=valuation_date,
            spot=spot,
            risk_free_rate=risk_free_rate,
            volatility=call_volatility,
            dividend_yield=dividend_yield,
        )

        put_value = self.put_position.market_value(
            valuation_date=valuation_date,
            spot=spot,
            risk_free_rate=risk_free_rate,
            volatility=put_volatility,
            dividend_yield=dividend_yield,
        )

        return call_value + put_value

    def pnl(
        self,
        valuation_date: pd.Timestamp,
        spot: float,
        risk_free_rate: float,
        call_volatility: float,
        put_volatility: float,
        dividend_yield: float = 0.0,
    ) -> float:
        """Calculate straddle mark-to-market P&L before hedging and fees."""
        return self.market_value(
            valuation_date=valuation_date,
            spot=spot,
            risk_free_rate=risk_free_rate,
            call_volatility=call_volatility,
            put_volatility=put_volatility,
            dividend_yield=dividend_yield,
        ) - self.entry_cost

    def combined_greeks(
        self,
        valuation_date: pd.Timestamp,
        spot: float,
        risk_free_rate: float,
        call_volatility: float,
        put_volatility: float,
        dividend_yield: float = 0.0,
    ) -> dict[str, float]:
        """Aggregate portfolio Greeks after contract multiplier and quantity."""
        call_result = self.call_position.price_and_greeks(
            valuation_date=valuation_date,
            spot=spot,
            risk_free_rate=risk_free_rate,
            volatility=call_volatility,
            dividend_yield=dividend_yield,
        )

        put_result = self.put_position.price_and_greeks(
            valuation_date=valuation_date,
            spot=spot,
            risk_free_rate=risk_free_rate,
            volatility=put_volatility,
            dividend_yield=dividend_yield,
        )

        multiplier = self.call_position.contract.multiplier
        quantity = self.quantity

        return {
            "delta": (
                call_result.delta + put_result.delta
            )
            * quantity
            * multiplier,
            "gamma": (
                call_result.gamma + put_result.gamma
            )
            * quantity
            * multiplier,
            "vega": (
                call_result.vega + put_result.vega
            )
            * quantity
            * multiplier,
            "theta": (
                call_result.theta + put_result.theta
            )
            * quantity
            * multiplier,
            "rho": (
                call_result.rho + put_result.rho
            )
            * quantity
            * multiplier,
        }


def build_long_atm_straddle(
    chain: pd.DataFrame,
    quantity: int = 1,
    multiplier: int = 100,
) -> LongStraddle:
    """
    Build a long straddle from a two-row aligned call-put option chain subset.

    Expected fields:
    valuation_date, expiry_date, option_type, strike, ask.
    Long option positions are entered at the ask price.
    """
    required_columns = {
        "valuation_date",
        "expiry_date",
        "option_type",
        "strike",
        "ask",
    }

    missing_columns = required_columns - set(chain.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if len(chain) != 2:
        raise ValueError(
            "Straddle construction requires exactly two option rows."
        )

    if quantity < 1:
        raise ValueError("quantity must be at least 1.")

    if multiplier < 1:
        raise ValueError("multiplier must be at least 1.")

    call_rows = chain.loc[
        chain["option_type"] == "call"
    ]
    put_rows = chain.loc[
        chain["option_type"] == "put"
    ]

    if len(call_rows) != 1 or len(put_rows) != 1:
        raise ValueError(
            "Straddle requires exactly one call and one put."
        )

    call_row = call_rows.iloc[0]
    put_row = put_rows.iloc[0]

    call_contract = OptionContract(
        option_type="call",
        strike=float(call_row["strike"]),
        expiry_date=pd.Timestamp(call_row["expiry_date"]),
        multiplier=multiplier,
    )

    put_contract = OptionContract(
        option_type="put",
        strike=float(put_row["strike"]),
        expiry_date=pd.Timestamp(put_row["expiry_date"]),
        multiplier=multiplier,
    )

    call_position = OptionPosition(
        contract=call_contract,
        quantity=quantity,
        entry_price=float(call_row["ask"]),
    )

    put_position = OptionPosition(
        contract=put_contract,
        quantity=quantity,
        entry_price=float(put_row["ask"]),
    )

    return LongStraddle(
        call_position=call_position,
        put_position=put_position,
    )
