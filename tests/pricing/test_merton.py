import math

from src.domain.enums import OptionType
from src.pricing.black_scholes import option_price
from src.pricing.merton import merton_mc_price, merton_series_price


def _params():
    return dict(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        volatility=0.2,
        risk_free_rate=0.04,
        dividend_yield=0.0,
    )


def test_merton_series_degenerates_to_black_scholes():
    params = _params()
    reference = option_price(
        **params,
        option_type="call",
    )
    series = merton_series_price(
        **params,
        option_type=OptionType.CALL,
        jump_intensity=0.0,
        jump_mean=0.0,
        jump_vol=0.0,
    )
    assert abs(series - reference) < 1e-9


def test_merton_mc_matches_series_within_ci():
    params = _params()
    series = merton_series_price(
        **params,
        option_type=OptionType.CALL,
        jump_intensity=0.5,
        jump_mean=-0.05,
        jump_vol=0.15,
    )
    mc = merton_mc_price(
        **params,
        option_type=OptionType.CALL,
        jump_intensity=0.5,
        jump_mean=-0.05,
        jump_vol=0.15,
        n_paths=60_000,
        seed=4,
    )
    assert abs(mc["price"] - series) < 3 * mc["standard_error"]
