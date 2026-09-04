import pandas as pd

from src.backtest.long_straddle_backtest import (
    _base_volatility_for_row,
    add_realized_volatility_features,
)
from src.market_data.synthetic_option_chain import (
    VolatilitySurfaceParameters,
    create_synthetic_option_chain,
)


def _build_data(periods: int = 420, shock_start: int | None = None):
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    prices = [
        100.0 + 0.02 * index + 2.0 * ((index % 7) - 3)
        for index in range(periods)
    ]
    if shock_start is not None:
        for index in range(shock_start, periods):
            prices[index] = prices[index] * 5.0
    return pd.DataFrame({"Close": prices}, index=dates)


def test_synthetic_chain_before_shock_is_unchanged():
    plain = _build_data()
    shocked = _build_data(shock_start=360)

    plain_features = add_realized_volatility_features(
        plain,
        price_column="Close",
        windows=(20, 60, 252),
    )
    shocked_features = add_realized_volatility_features(
        shocked,
        price_column="Close",
        windows=(20, 60, 252),
    )

    evaluation_date = plain.index[340]
    plain_row = plain_features.loc[evaluation_date]
    shocked_row = shocked_features.loc[evaluation_date]

    plain_chain = create_synthetic_option_chain(
        valuation_date=evaluation_date,
        spot=float(plain_row["Close"]),
        base_volatility=_base_volatility_for_row(plain_row),
        risk_free_rate=0.04,
        dividend_yield=0.012,
        days_to_expiry=(30,),
        strike_multipliers=(0.95, 1.0, 1.05),
        parameters=VolatilitySurfaceParameters(),
    )
    shocked_chain = create_synthetic_option_chain(
        valuation_date=evaluation_date,
        spot=float(shocked_row["Close"]),
        base_volatility=_base_volatility_for_row(shocked_row),
        risk_free_rate=0.04,
        dividend_yield=0.012,
        days_to_expiry=(30,),
        strike_multipliers=(0.95, 1.0, 1.05),
        parameters=VolatilitySurfaceParameters(),
    )

    plain_calls = plain_chain[
        plain_chain["option_type"] == "call"
    ]["implied_volatility"].reset_index(drop=True)
    shocked_calls = shocked_chain[
        shocked_chain["option_type"] == "call"
    ]["implied_volatility"].reset_index(drop=True)
    plain_puts = plain_chain[
        plain_chain["option_type"] == "put"
    ]["implied_volatility"].reset_index(drop=True)
    shocked_puts = shocked_chain[
        shocked_chain["option_type"] == "put"
    ]["implied_volatility"].reset_index(drop=True)

    assert plain_calls.equals(shocked_calls)
    assert plain_puts.equals(shocked_puts)
