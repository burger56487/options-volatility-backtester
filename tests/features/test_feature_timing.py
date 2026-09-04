import pandas as pd

from src.features.audit import assert_no_feature_lookahead
from src.features.feature_frame import (
    build_realised_volatility_features,
)


def test_realised_volatility_is_lagged():
    dates = pd.bdate_range("2024-01-01", periods=300)
    prices = pd.DataFrame(
        {
            "date": dates,
            "adjusted_close": [
                100 + index * 0.1 for index in range(len(dates))
            ],
        }
    )
    features = build_realised_volatility_features(prices)
    valid = features.dropna(subset=["rv_20", "rv_252"])
    assert (
        valid["observation_end_date"] < valid["signal_date"]
    ).all()
    assert_no_feature_lookahead(valid)
