from __future__ import annotations

from dataclasses import dataclass, field
from math import copysign

import pandas as pd


@dataclass(frozen=True)
class UnderlyingTransactionCostModel:
    """
    Cost model for underlying-asset hedge trades.

    commission_per_share:
        Fixed commission charged per share traded.

    slippage_bps:
        Half-spread/slippage charge in basis points of trade notional.
        For example, 1.0 means 1 basis point = 0.01%.
    """

    commission_per_share: float = 0.0
    slippage_bps: float = 1.0

    def __post_init__(self) -> None:
        if self.commission_per_share < 0:
            raise ValueError(
                "commission_per_share must be non-negative."
            )

        if self.slippage_bps < 0:
            raise ValueError(
                "slippage_bps must be non-negative."
            )

    def transaction_cost(
        self,
        quantity: float,
        spot: float,
    ) -> float:
        """Calculate positive transaction costs for one hedge trade."""
        if spot <= 0:
            raise ValueError("spot must be positive.")

        absolute_quantity = abs(quantity)

        commission = (
            absolute_quantity * self.commission_per_share
        )

        slippage = (
            absolute_quantity
            * spot
            * self.slippage_bps
            / 10_000.0
        )

        return commission + slippage


@dataclass(frozen=True)
class HedgeTrade:
    """One underlying hedge transaction."""

    trade_date: pd.Timestamp
    quantity: float
    reference_spot: float
    execution_price: float
    transaction_cost: float
    pre_trade_position: float
    post_trade_position: float
    option_delta: float
    post_hedge_delta: float

    @property
    def notional(self) -> float:
        """Absolute reference notional traded."""
        return abs(self.quantity * self.reference_spot)


@dataclass
class DeltaHedger:
    """
    Maintain a hedge position in the underlying asset.

    The target underlying position is the negative option portfolio Delta.
    Trades only occur when the required adjustment exceeds delta_threshold.
    """

    cost_model: UnderlyingTransactionCostModel = field(
        default_factory=UnderlyingTransactionCostModel
    )
    delta_threshold: float = 0.0
    allow_fractional_shares: bool = True
    position: float = 0.0
    cash: float = 0.0
    cumulative_turnover: float = 0.0
    cumulative_transaction_costs: float = 0.0
    trades: list[HedgeTrade] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.delta_threshold < 0:
            raise ValueError(
                "delta_threshold must be non-negative."
            )

    @staticmethod
    def target_position(
        option_delta: float,
        allow_fractional_shares: bool = True,
    ) -> float:
        """
        Calculate target underlying hedge position.

        A positive option Delta requires a short underlying hedge position.
        """
        target = -option_delta

        if allow_fractional_shares:
            return target

        return float(round(target))

    @staticmethod
    def _execution_price(
        spot: float,
        quantity: float,
        slippage_bps: float,
    ) -> float:
        """
        Calculate adverse execution price.

        Buy trades execute above spot; sell trades execute below spot.
        """
        if quantity == 0:
            return spot

        slippage_fraction = slippage_bps / 10_000.0

        return spot * (
            1.0
            + copysign(
                slippage_fraction,
                quantity,
            )
        )

    def rebalance(
        self,
        trade_date: pd.Timestamp,
        spot: float,
        option_delta: float,
    ) -> HedgeTrade | None:
        """
        Rebalance the underlying hedge to offset option portfolio Delta.

        Returns a HedgeTrade when a trade occurs; otherwise returns None.
        """
        if pd.isna(trade_date):
            raise ValueError("trade_date must not be missing.")

        if spot <= 0:
            raise ValueError("spot must be positive.")

        target_position = self.target_position(
            option_delta=option_delta,
            allow_fractional_shares=self.allow_fractional_shares,
        )

        trade_quantity = target_position - self.position

        if abs(trade_quantity) <= self.delta_threshold:
            return None

        execution_price = self._execution_price(
            spot=spot,
            quantity=trade_quantity,
            slippage_bps=self.cost_model.slippage_bps,
        )

        transaction_cost = self.cost_model.transaction_cost(
            quantity=trade_quantity,
            spot=spot,
        )

        pre_trade_position = self.position
        post_trade_position = (
            pre_trade_position + trade_quantity
        )

        # Buying underlying consumes cash; selling underlying receives cash.
        self.cash -= trade_quantity * execution_price
        self.cash -= transaction_cost

        self.position = post_trade_position
        self.cumulative_turnover += abs(
            trade_quantity * spot
        )
        self.cumulative_transaction_costs += transaction_cost

        post_hedge_delta = option_delta + post_trade_position

        trade = HedgeTrade(
            trade_date=pd.Timestamp(trade_date),
            quantity=trade_quantity,
            reference_spot=spot,
            execution_price=execution_price,
            transaction_cost=transaction_cost,
            pre_trade_position=pre_trade_position,
            post_trade_position=post_trade_position,
            option_delta=option_delta,
            post_hedge_delta=post_hedge_delta,
        )

        self.trades.append(trade)

        return trade

    def market_value(
        self,
        spot: float,
    ) -> float:
        """Calculate marked-to-market value of the hedge position."""
        if spot <= 0:
            raise ValueError("spot must be positive.")

        return self.position * spot

    def total_equity(
        self,
        spot: float,
    ) -> float:
        """
        Calculate total hedge-account value: cash plus underlying position value.
        """
        return self.cash + self.market_value(spot)

    def turnover_ratio(
        self,
        initial_portfolio_value: float,
    ) -> float:
        """
        Calculate cumulative hedge turnover relative to initial portfolio value.
        """
        if initial_portfolio_value <= 0:
            raise ValueError(
                "initial_portfolio_value must be positive."
            )

        return (
            self.cumulative_turnover
            / initial_portfolio_value
        )

    def trade_log(self) -> pd.DataFrame:
        """Return recorded hedge transactions as a DataFrame."""
        columns = [
            "trade_date",
            "quantity",
            "reference_spot",
            "execution_price",
            "transaction_cost",
            "pre_trade_position",
            "post_trade_position",
            "option_delta",
            "post_hedge_delta",
            "notional",
        ]

        if not self.trades:
            return pd.DataFrame(columns=columns)

        records = [
            {
                "trade_date": trade.trade_date,
                "quantity": trade.quantity,
                "reference_spot": trade.reference_spot,
                "execution_price": trade.execution_price,
                "transaction_cost": trade.transaction_cost,
                "pre_trade_position": trade.pre_trade_position,
                "post_trade_position": trade.post_trade_position,
                "option_delta": trade.option_delta,
                "post_hedge_delta": trade.post_hedge_delta,
                "notional": trade.notional,
            }
            for trade in self.trades
        ]

        return pd.DataFrame(records)
