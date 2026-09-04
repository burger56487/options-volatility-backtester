"""Same-seed runs must reproduce identical results."""

from src.domain.enums import OptionType
from src.pricing.heston import heston_mc_price
from src.pricing.merton import merton_mc_price


def test_heston_mc_reproducible_with_seed():
    params = dict(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        kappa=1.0,
        theta=0.04,
        sigma_v=0.2,
        rho=-0.4,
        v0=0.04,
        risk_free_rate=0.04,
        option_type=OptionType.CALL,
        n_paths=20_000,
        n_steps=30,
        seed=9,
    )
    first = heston_mc_price(**params)["price"]
    second = heston_mc_price(**params)["price"]
    assert first == second


def test_merton_mc_reproducible_with_seed():
    params = dict(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        volatility=0.2,
        risk_free_rate=0.04,
        option_type=OptionType.CALL,
        jump_intensity=0.5,
        jump_mean=-0.05,
        jump_vol=0.15,
        n_paths=30_000,
        seed=11,
    )
    first = merton_mc_price(**params)["price"]
    second = merton_mc_price(**params)["price"]
    assert first == second
