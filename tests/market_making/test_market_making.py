import numpy as np

from src.market_making.policies import (
    avellaneda_stoikov_offsets,
    fixed_quote_offsets,
    inventory_skew_offsets,
)
from src.market_making.simulator import (
    fill_probability,
    market_making_metrics,
    simulate_quotes,
)


def test_fixed_offsets_symmetric():
    bid, ask = fixed_quote_offsets(0.1)
    assert bid == -0.1
    assert ask == 0.1


def test_inventory_skew_shifts_both_quotes_down_when_long():
    bid, ask = inventory_skew_offsets(0.1, inventory=1.0, max_inventory=1.0)
    assert ask < 0.1
    assert bid < 0.0


def test_inventory_skew_raises_quotes_when_short():
    bid, ask = inventory_skew_offsets(
        0.1, inventory=-1.0, max_inventory=1.0
    )
    assert ask > 0.1
    assert bid > -0.1


def test_as_spread_widens_with_volatility():
    low = avellaneda_stoikov_offsets(0.0, 1.0, 0.1, 0.01, 10.0)
    high = avellaneda_stoikov_offsets(0.0, 1.0, 0.3, 0.01, 10.0)
    assert (high[1] - high[0]) > (low[1] - low[0])


def test_simulator_metrics_and_fill_probability():
    rng = np.random.default_rng(2)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.1, 100))
    result = simulate_quotes(mid, half_spread=0.05, seed=2)
    metrics = market_making_metrics(result)
    assert "terminal_pnl" in metrics
    assert 0.0 <= fill_probability(0.0, 0.05) <= 1.0
    assert fill_probability(1.0, 0.05) < fill_probability(0.0, 0.05)
