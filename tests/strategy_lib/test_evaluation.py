from src.strategy_lib.evaluation import paper_strategy_pnl
from src.strategy_lib.strategies import long_straddle
from src.strategy_lib.strategies2 import risk_reversal


def test_straddle_paper_pnl_negative_without_move():
    pnl = paper_strategy_pnl(
        long_straddle(30),
        entry_spot=100.0,
        exit_spot=100.0,
        days_held=5,
        risk_free_rate=0.04,
        volatility=0.2,
    )
    assert pnl < 0  # theta decay dominates at flat spot


def test_risk_reversal_direction():
    down_pnl = paper_strategy_pnl(
        risk_reversal(30),
        entry_spot=100.0,
        exit_spot=96.0,
        days_held=10,
        risk_free_rate=0.04,
        volatility=0.2,
    )
    up_pnl = paper_strategy_pnl(
        risk_reversal(30),
        entry_spot=100.0,
        exit_spot=104.0,
        days_held=10,
        risk_free_rate=0.04,
        volatility=0.2,
    )
    assert down_pnl > up_pnl
