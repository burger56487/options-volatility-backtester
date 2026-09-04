from src.execution.latency import sample_latency_seconds
from src.execution.slicing import (
    opportunity_cost_estimate,
    slice_order,
)


def test_slice_order_respects_participation_and_total():
    slices = slice_order(
        quantity=1200,
        volume=10_000,
        max_participation_rate=0.05,
    )
    assert sum(slices) == 1200
    assert all(take <= 500 for take in slices)
    assert len(slices) >= 3


def test_opportunity_cost_is_non_negative():
    assert opportunity_cost_estimate(100, 200.0, 1.0) == 2.0


def test_latency_sampling_is_bounded_and_deterministic():
    first = sample_latency_seconds(0.05, 0.10, seed=7)
    second = sample_latency_seconds(0.05, 0.10, seed=7)
    assert first == second
    assert 0.05 <= first <= 0.15
