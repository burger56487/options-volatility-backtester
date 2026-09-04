"""Python single-thread Black-Scholes loop (baseline for C++ speedup)."""

from __future__ import annotations

import math
import time


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_call(S, K, T, r, q, sigma):
    sigma_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def main() -> None:
    n = 1_000_000
    s, k, t, r, q, sigma = 100.0, 100.0, 0.5, 0.04, 0.01, 0.25
    start = time.perf_counter()
    total = 0.0
    for _ in range(n):
        total += bs_call(s, k, t, r, q, sigma)
    seconds = time.perf_counter() - start
    print(f"paths={n} seconds={seconds:.6f} avg={total / n:.10f}")


if __name__ == "__main__":
    main()
