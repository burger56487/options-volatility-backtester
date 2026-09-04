from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.market_data.realized_volatility import (
    add_realized_volatility_features,
)
from src.market_data.synthetic_option_chain import (
    VolatilitySurfaceParameters,
    blended_realised_volatility,
    create_synthetic_option_chain,
    select_atm_straddle,
    synthetic_implied_volatility,
)
from src.strategy.delta_hedging import (
    DeltaHedger,
    UnderlyingTransactionCostModel,
)
from src.experiment_meta import experiment_metadata
from src.strategy.long_straddle import (
    LongStraddle,
    build_long_atm_straddle,
)


@dataclass(frozen=True)
class LongStraddleBacktestConfig:
    """Configuration for one delta-hedged long-straddle backtest."""

    days_to_expiry: int = 30
    quantity: int = 1
    multiplier: int = 100
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.012
    delta_threshold: float = 1.0
    allow_fractional_shares: bool = False
    commission_per_share: float = 0.005
    underlying_slippage_bps: float = 1.0
    volatility_windows: tuple[int, int, int] = (20, 60, 252)

    def __post_init__(self) -> None:
        if self.days_to_expiry < 1:
            raise ValueError(
                "days_to_expiry must be at least 1."
            )

        if self.quantity < 1:
            raise ValueError("quantity must be at least 1.")

        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1.")

        if self.delta_threshold < 0:
            raise ValueError(
                "delta_threshold must be non-negative."
            )


@dataclass(frozen=True)
class LongStraddleBacktestResult:
    """Artifacts and summary statistics from a single-trade backtest."""

    equity_curve: pd.DataFrame
    hedge_trades: pd.DataFrame
    summary: dict[str, float | int | str]


def _validate_price_data(
    price_data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and sort input daily price data."""
    required_columns = {"Close"}

    if price_data.empty:
        raise ValueError("price_data must not be empty.")

    missing_columns = required_columns - set(price_data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if not isinstance(price_data.index, pd.DatetimeIndex):
        raise TypeError(
            "price_data index must be a DatetimeIndex."
        )

    if (price_data["Close"].dropna() <= 0).any():
        raise ValueError("Close prices must be positive.")

    result = price_data.copy()
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()

    return result


def _base_volatility_for_row(
    row: pd.Series,
) -> float:
    """Build blended realised volatility from one featured data row."""
    return blended_realised_volatility(
        realized_vol_20d=float(row["realised_vol_20d"]),
        realized_vol_60d=float(row["realised_vol_60d"]),
        realized_vol_252d=float(row["realised_vol_252d"]),
    )


def _daily_option_volatilities(
    spot: float,
    strike: float,
    time_to_expiry: float,
    base_volatility: float,
    surface_parameters: VolatilitySurfaceParameters,
) -> tuple[float, float]:
    """Return synthetic call and put implied volatilities."""
    call_volatility = synthetic_implied_volatility(
        spot=spot,
        strike=strike,
        time_to_expiry=max(time_to_expiry, 1 / 365),
        base_volatility=base_volatility,
        option_type="call",
        parameters=surface_parameters,
    )

    put_volatility = synthetic_implied_volatility(
        spot=spot,
        strike=strike,
        time_to_expiry=max(time_to_expiry, 1 / 365),
        base_volatility=base_volatility,
        option_type="put",
        parameters=surface_parameters,
    )

    return call_volatility, put_volatility


def run_long_straddle_backtest(
    price_data: pd.DataFrame,
    entry_date: pd.Timestamp,
    config: LongStraddleBacktestConfig = (
        LongStraddleBacktestConfig()
    ),
    surface_parameters: VolatilitySurfaceParameters = (
        VolatilitySurfaceParameters()
    ),
) -> LongStraddleBacktestResult:
    """
    Backtest one ATM long straddle with delta hedging.

    The strategy:
    1. Uses the entry-date SPY close and blended realised volatility;
    2. Generates a synthetic ATM option chain;
    3. Buys Call and Put at their respective ask prices;
    4. Revalues the options each subsequent available trading day;
    5. Rebalances underlying Delta hedge when the adjustment exceeds threshold;
    6. Exits at expiry or the final available day, marking options at intrinsic
       value if expired.

    The simulation uses adjusted daily Close prices and simplified synthetic
    volatility-surface assumptions. It is intended for educational research.
    """
    data = _validate_price_data(price_data)

    if pd.isna(entry_date):
        raise ValueError("entry_date must not be missing.")

    entry_date = pd.Timestamp(entry_date)

    if entry_date not in data.index:
        raise ValueError(
            "entry_date must exist in the price-data index."
        )

    featured_data = add_realized_volatility_features(
        data=data,
        price_column="Close",
        windows=config.volatility_windows,
    )

    volatility_columns = [
        f"realised_vol_{window}d"
        for window in config.volatility_windows
    ]

    entry_row = featured_data.loc[entry_date]

    if entry_row[volatility_columns].isna().any():
        raise ValueError(
            "Insufficient price history before entry_date for "
            "realised-volatility estimation."
        )

    entry_spot = float(entry_row["Close"])
    entry_base_volatility = _base_volatility_for_row(entry_row)

    entry_chain = create_synthetic_option_chain(
        valuation_date=entry_date,
        spot=entry_spot,
        base_volatility=entry_base_volatility,
        risk_free_rate=config.risk_free_rate,
        dividend_yield=config.dividend_yield,
        days_to_expiry=(config.days_to_expiry,),
        strike_multipliers=(0.95, 1.0, 1.05),
        parameters=surface_parameters,
    )

    atm_chain = select_atm_straddle(
        chain=entry_chain,
        days_to_expiry=config.days_to_expiry,
    )

    straddle = build_long_atm_straddle(
        chain=atm_chain,
        quantity=config.quantity,
        multiplier=config.multiplier,
    )

    cost_model = UnderlyingTransactionCostModel(
        commission_per_share=config.commission_per_share,
        slippage_bps=config.underlying_slippage_bps,
    )

    hedger = DeltaHedger(
        cost_model=cost_model,
        delta_threshold=config.delta_threshold,
        allow_fractional_shares=config.allow_fractional_shares,
    )

    expiry_date = straddle.expiry_date

    trade_data = featured_data.loc[
        (featured_data.index >= entry_date)
        & (featured_data.index <= expiry_date)
    ].copy()

    if trade_data.empty:
        raise ValueError(
            "No price observations are available for the trade period."
        )

    records: list[dict[str, float | int | str | pd.Timestamp]] = []

    for valuation_date, row in trade_data.iterrows():
        spot = float(row["Close"])
        base_volatility = _base_volatility_for_row(row)

        time_to_expiry = max(
            straddle.call_position.time_to_expiry(
                valuation_date
            ),
            0.0,
        )

        call_volatility, put_volatility = (
            _daily_option_volatilities(
                spot=spot,
                strike=straddle.strike,
                time_to_expiry=time_to_expiry,
                base_volatility=base_volatility,
                surface_parameters=surface_parameters,
            )
        )

        option_value_before_hedge = straddle.market_value(
            valuation_date=valuation_date,
            spot=spot,
            risk_free_rate=config.risk_free_rate,
            call_volatility=call_volatility,
            put_volatility=put_volatility,
            dividend_yield=config.dividend_yield,
        )

        greeks = straddle.combined_greeks(
            valuation_date=valuation_date,
            spot=spot,
            risk_free_rate=config.risk_free_rate,
            call_volatility=call_volatility,
            put_volatility=put_volatility,
            dividend_yield=config.dividend_yield,
        )

        hedge_trade = hedger.rebalance(
            trade_date=valuation_date,
            spot=spot,
            option_delta=greeks["delta"],
        )

        hedge_value = hedger.total_equity(spot)

        total_equity = (
            option_value_before_hedge
            + hedge_value
            - straddle.entry_cost
        )

        records.append(
            {
                "date": valuation_date,
                "spot": spot,
                "base_volatility": base_volatility,
                "call_implied_volatility": call_volatility,
                "put_implied_volatility": put_volatility,
                "option_value": option_value_before_hedge,
                "option_pnl": (
                    option_value_before_hedge
                    - straddle.entry_cost
                ),
                "option_delta": greeks["delta"],
                "gamma": greeks["gamma"],
                "vega": greeks["vega"],
                "theta": greeks["theta"],
                "rho": greeks["rho"],
                "hedge_position": hedger.position,
                "hedge_cash": hedger.cash,
                "hedge_equity": hedge_value,
                "hedge_trade_quantity": (
                    0.0
                    if hedge_trade is None
                    else hedge_trade.quantity
                ),
                "hedge_transaction_cost": (
                    0.0
                    if hedge_trade is None
                    else hedge_trade.transaction_cost
                ),
                "post_hedge_delta": (
                    greeks["delta"] + hedger.position
                ),
                "cumulative_turnover": hedger.cumulative_turnover,
                "cumulative_hedge_costs": (
                    hedger.cumulative_transaction_costs
                ),
                "total_pnl": total_equity,
            }
        )

    equity_curve = pd.DataFrame(records).set_index("date")

    hedge_trades = hedger.trade_log()

    starting_cost = straddle.entry_cost
    final_pnl = float(equity_curve["total_pnl"].iloc[-1])
    max_equity = equity_curve["total_pnl"].cummax()
    drawdown = equity_curve["total_pnl"] - max_equity
    max_drawdown = float(drawdown.min())

    summary: dict[str, float | int | str] = {
        "entry_date": str(entry_date.date()),
        "expiry_date": str(expiry_date.date()),
        "entry_spot": entry_spot,
        "strike": straddle.strike,
        "entry_cost": starting_cost,
        "final_spot": float(equity_curve["spot"].iloc[-1]),
        "final_pnl": final_pnl,
        "max_drawdown": max_drawdown,
        "number_of_hedge_trades": int(len(hedge_trades)),
        "cumulative_hedge_turnover": (
            hedger.cumulative_turnover
        ),
        "cumulative_hedge_costs": (
            hedger.cumulative_transaction_costs
        ),
        "hedge_turnover_ratio": hedger.turnover_ratio(
            initial_portfolio_value=starting_cost
        ),
        "final_option_value": float(
            equity_curve["option_value"].iloc[-1]
        ),
        "final_hedge_equity": float(
            equity_curve["hedge_equity"].iloc[-1]
        ),
        **experiment_metadata(),
    }

    return LongStraddleBacktestResult(
        equity_curve=equity_curve,
        hedge_trades=hedge_trades,
        summary=summary,
    )


def save_backtest_result(
    result: LongStraddleBacktestResult,
    output_directory: str | Path,
    prefix: str = "single_trade",
) -> None:
    """Save equity curve, hedge trades, and summary artifacts."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    result.equity_curve.to_csv(
        output_path / f"{prefix}_equity_curve.csv"
    )

    result.hedge_trades.to_csv(
        output_path / f"{prefix}_hedge_trades.csv",
        index=False,
    )

    summary_frame = pd.DataFrame(
        [result.summary]
    )

    summary_frame.to_json(
        output_path / f"{prefix}_summary.json",
        orient="records",
        indent=2,
    )
