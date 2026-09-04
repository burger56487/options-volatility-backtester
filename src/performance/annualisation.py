"""Annualisation helpers."""

from __future__ import annotations


def annualised_return(
    total_return: float,
    n_observations: int,
    periods_per_year: float = 252.0,
) -> float:
    if n_observations <= 0:
        return 0.0
    if total_return <= -1.0:
        return -1.0
    return (
        1.0 + total_return
    ) ** (periods_per_year / n_observations) - 1.0
