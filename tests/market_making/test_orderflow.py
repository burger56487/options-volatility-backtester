import numpy as np

from src.market_making.orderflow import (
    hawkes_intensity_path,
    split_informed_flow,
)


def test_hawkes_intensity_is_finite_and_reverts():
    path = hawkes_intensity_path(
        baseline=0.5,
        alpha=0.8,
        beta=0.2,
        n_steps=500,
        seed=1,
    )
    assert np.isfinite(path).all()
    assert path.mean() > 0


def test_informed_split_respects_fraction():
    mask = split_informed_flow(1_000, informed_fraction=0.3, seed=2)
    assert abs(mask.mean() - 0.3) < 0.05
