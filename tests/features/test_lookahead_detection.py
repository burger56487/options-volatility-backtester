import pandas as pd
import pytest

from src.features.audit import (
    assert_no_feature_lookahead,
    audit_feature_timing,
)


def test_same_day_observation_is_rejected():
    dataframe = pd.DataFrame(
        {
            "signal_date": ["2025-01-10"],
            "observation_end_date": ["2025-01-10"],
        }
    )
    with pytest.raises(ValueError):
        assert_no_feature_lookahead(dataframe)


def test_future_observation_is_rejected():
    dataframe = pd.DataFrame(
        {
            "signal_date": ["2025-01-10"],
            "observation_end_date": ["2025-01-11"],
        }
    )
    assert len(audit_feature_timing(dataframe)) == 1
