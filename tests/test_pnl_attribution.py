import numpy as np
import pandas as pd

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
    run_long_straddle_backtest,
)
from src.backtest.pnl_attribution import (
    attribute_daily_pnl,
    attribution_summary,
)


def _create_price_data(periods: int = 320) -> pd.DataFrame:
    dates = pd.date_range(
        "2023-01-02",
        periods=periods,
        freq="B",
    )
    prices = [
        100.0
        + 0.04 * index
        + 3.0 * ((index % 11) - 5)
        for index in range(periods)
    ]
    return pd.DataFrame({"Close": prices}, index=dates)


def _run_trade() -> tuple:
    data = _create_price_data()
    entry_date = data.index[260]
    result = run_long_straddle_backtest(
        price_data=data,
        entry_date=entry_date,
        config=LongStraddleBacktestConfig(days_to_expiry=10),
    )
    return data, result


def test_attribution_columns_and_identity():
    _, result = _run_trade()
    attribution = attribute_daily_pnl(result.equity_curve)

    expected_columns = {
        "actual_pnl_change",
        "delta_contribution",
        "gamma_contribution",
        "vega_contribution",
        "theta_contribution",
        "rho_contribution",
        "cost_contribution",
        "model_contribution",
        "residual",
    }
    assert expected_columns.issubset(attribution.columns)

    valid = attribution.iloc[1:]
    assert np.allclose(
        valid["model_contribution"] + valid["residual"],
        valid["actual_pnl_change"],
    )


def test_attribution_summary_totals_match_equity_curve():
    _, result = _run_trade()
    attribution = attribute_daily_pnl(result.equity_curve)
    summary = attribution_summary(attribution)

    first_total = float(result.equity_curve["total_pnl"].iloc[0])
    last_total = float(result.equity_curve["total_pnl"].iloc[-1])
    assert (
        abs(summary["actual_pnl_total"] - (last_total - first_total))
        < 1e-6
    )

    assert 0.0 <= summary["abs_residual_ratio"] <= 1.5
