"""Merton jump-diffusion pricing: analytic series and Monte Carlo."""

from __future__ import annotations

import math

import numpy as np

from src.domain.enums import OptionType
from src.pricing.black_scholes import option_price


def _compensator(jump_mean: float, jump_vol: float) -> float:
    return math.exp(jump_mean + 0.5 * jump_vol**2) - 1.0


def merton_series_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    option_type: OptionType,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    dividend_yield: float = 0.0,
    max_terms: int = 60,
    tolerance: float = 1e-12,
) -> float:
    """Price a European option via the Poisson-weighted Black-Scholes series."""
    k = _compensator(jump_mean, jump_vol)
    lam_prime = jump_intensity * (1.0 + k)
    total = 0.0
    last_weight = 0.0
    poisson_mean = lam_prime * time_to_expiry
    for n in range(max_terms):
        weight = math.exp(-lam_prime * time_to_expiry) * (
            (lam_prime * time_to_expiry) ** n
        ) / math.factorial(n)
        last_weight = weight
        # Stop only in the right tail (n past the Poisson mean), where the
        # probability mass is monotonically decreasing.  Before the mode the
        # weight can be below tolerance while the bulk of the mass still lies
        # ahead (e.g. high jump intensity), so a left-tail break would
        # silently truncate the whole series.
        if (
            weight < tolerance
            and n > 2
            and n > poisson_mean
        ):
            break
        rate_n = (
            risk_free_rate
            - jump_intensity * k
            + n * math.log(1.0 + k) / time_to_expiry
        )
        vol_n = math.sqrt(
            volatility**2
            + n * jump_vol**2 / time_to_expiry
        )
        total += weight * option_price(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=rate_n,
            volatility=vol_n,
            option_type=option_type.value,
            dividend_yield=dividend_yield,
        )
    else:
        if last_weight >= tolerance:
            raise ValueError(
                "Merton series reached max_terms while the Poisson weight "
                f"({last_weight:.3e}) is still above tolerance "
                f"({tolerance:.1e}); increase max_terms or reduce "
                "jump_intensity * time_to_expiry."
            )
    return float(total)


def merton_mc_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    option_type: OptionType,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    dividend_yield: float = 0.0,
    n_paths: int = 60_000,
    seed: int | None = None,
) -> dict:
    """Price via compound-Poisson Monte Carlo."""
    rng = np.random.default_rng(seed)
    k = _compensator(jump_mean, jump_vol)
    diffusion = volatility * math.sqrt(time_to_expiry)
    z = rng.standard_normal(n_paths)
    jump_counts = rng.poisson(
        jump_intensity * time_to_expiry,
        size=n_paths,
    )
    total_jump = np.zeros(n_paths)
    max_jumps = int(jump_counts.max())
    for _ in range(max_jumps):
        active = jump_counts > 0
        total_jump[active] += rng.normal(
            jump_mean,
            jump_vol,
            size=int(active.sum()),
        )
        jump_counts = np.maximum(jump_counts - 1, 0)
    drift = (
        risk_free_rate
        - dividend_yield
        - jump_intensity * k
        - 0.5 * volatility**2
    ) * time_to_expiry
    terminal = spot * np.exp(
        drift + diffusion * z + total_jump
    )
    discount = math.exp(-risk_free_rate * time_to_expiry)
    if option_type == OptionType.CALL:
        payoffs = np.maximum(terminal - strike, 0.0)
    else:
        payoffs = np.maximum(strike - terminal, 0.0)
    prices = discount * payoffs
    price = float(np.mean(prices))
    se = float(np.std(prices, ddof=1) / math.sqrt(len(prices)))
    return {
        "price": price,
        "standard_error": se,
        "ci_low": price - 1.96 * se,
        "ci_high": price + 1.96 * se,
    }
