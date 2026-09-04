from pathlib import Path

import pandas as pd

from src.market_data.pipeline import (
    underlying_clean_to_price_frame,
)


def test_clean_underlying_loads_as_close_frame(tmp_path: Path):
    path = tmp_path / "clean.csv"
    pd.DataFrame(
        {
            "trade_date": ["2026-01-02", "2026-01-05"],
            "adjusted_close": [102.0, 103.5],
        }
    ).to_csv(path, index=False)
    frame = underlying_clean_to_price_frame(path)
    assert list(frame.columns) == ["Close"]
    assert frame.index[0] == pd.Timestamp("2026-01-02")
