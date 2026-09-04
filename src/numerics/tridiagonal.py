"""Thomas algorithm for tridiagonal systems."""

from __future__ import annotations

from typing import Sequence


def solve_tridiagonal(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> list[float]:
    """Solve A x = rhs for a tridiagonal A (no pivoting)."""
    n = len(diagonal)
    if not (len(lower) == n - 1 and len(upper) == n - 1 and len(rhs) == n):
        raise ValueError("Tridiagonal dimensions are inconsistent.")
    if n == 0:
        return []

    c_prime = [0.0] * n
    d_prime = [0.0] * n
    c_prime[0] = upper[0] / diagonal[0]
    d_prime[0] = rhs[0] / diagonal[0]

    for i in range(1, n):
        denominator = diagonal[i] - lower[i - 1] * c_prime[i - 1]
        if abs(denominator) < 1e-14:
            raise ValueError("Tridiagonal matrix is singular.")
        if i < n - 1:
            c_prime[i] = upper[i] / denominator
        d_prime[i] = (
            rhs[i] - lower[i - 1] * d_prime[i - 1]
        ) / denominator

    solution = [0.0] * n
    solution[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    return solution
