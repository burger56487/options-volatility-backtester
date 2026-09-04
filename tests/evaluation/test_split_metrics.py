import pandas as pd

from src.evaluation.runner import split_metrics


def test_small_sample_is_flagged_and_not_selectable():
    trades = pd.DataFrame(
        {
            "trade_return": [0.01, 0.02, -0.01],
            "final_pnl": [100.0, 200.0, -100.0],
            "max_drawdown": [-0.01, -0.02, -0.03],
            "hedge_turnover_ratio": [0.5, 0.5, 0.5],
        }
    )
    metrics = split_metrics(trades, trades_per_year=8.4)
    assert metrics["insufficient_sample"] is True
    assert metrics["selection_score"] == float("-inf")


def test_adequate_sample_is_selectable():
    trades = pd.DataFrame(
        {
            "trade_return": [0.01] * 6,
            "final_pnl": [100.0] * 6,
            "max_drawdown": [-0.01] * 6,
            "hedge_turnover_ratio": [0.5] * 6,
        }
    )
    metrics = split_metrics(trades, trades_per_year=8.4)
    assert metrics["insufficient_sample"] is False
    assert metrics["selection_score"] != float("-inf")
