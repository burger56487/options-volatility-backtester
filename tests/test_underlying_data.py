from pathlib import Path

import pandas as pd
import pytest

from src.market_data.underlying_data import (
    add_return_features,
    clean_price_data,
    load_price_data,
    save_price_data,
    validate_price_data,
)


def create_valid_price_data() -> pd.DataFrame:
    index = pd.to_datetime(
        [
            "2025-01-02",
            "2025-01-03",
            "2025-01-06",
        ]
    )
    index.name = "Date"

    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
            "Volume": [1_000_000, 1_100_000, 900_000],
        },
        index=index,
    )



def test_valid_data_has_clean_quality_report():
    data = create_valid_price_data()

    report = validate_price_data(data)

    assert report.observations == 3
    assert report.duplicate_dates == 0
    assert report.missing_values == 0
    assert report.non_positive_prices == 0
    assert report.invalid_high_low_rows == 0
    assert report.negative_volume_rows == 0


def test_missing_required_columns_raise_error():
    data = create_valid_price_data().drop(columns=["Volume"])

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_price_data(data)


def test_non_datetime_index_raises_error():
    data = create_valid_price_data().reset_index(drop=True)

    with pytest.raises(
        TypeError,
        match="DatetimeIndex",
    ):
        validate_price_data(data)


def test_invalid_ohlc_relationship_is_detected():
    data = create_valid_price_data()
    data.loc[data.index[0], "High"] = 98.0

    report = validate_price_data(data)

    assert report.invalid_high_low_rows == 1


def test_clean_data_rejects_missing_values():
    data = create_valid_price_data()
    data.loc[data.index[0], "Close"] = None

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        clean_price_data(data)


def test_clean_data_removes_duplicate_dates_and_sorts():
    data = create_valid_price_data()

    duplicate = data.iloc[[0]].copy()
    duplicate.iloc[0, duplicate.columns.get_loc("High")] = 106.0
    duplicate.iloc[0, duplicate.columns.get_loc("Close")] = 105.0

    unsorted_with_duplicate = pd.concat(
        [data.iloc[[2]], duplicate, data.iloc[[1]], data.iloc[[0]]]
    )

    cleaned = clean_price_data(unsorted_with_duplicate)

    assert len(cleaned) == 3
    assert cleaned.index.is_monotonic_increasing
    assert cleaned.loc[pd.Timestamp("2025-01-02"), "Close"] == 101.0



def test_return_features_are_calculated():
    data = create_valid_price_data()

    result = add_return_features(data)

    assert pd.isna(result.iloc[0]["simple_return"])
    assert result.iloc[1]["simple_return"] == pytest.approx(
        (102.0 / 101.0) - 1.0
    )
    assert result.iloc[2]["log_return"] == pytest.approx(
        __import__("math").log(103.0 / 102.0)
    )


def test_invalid_price_column_raises_error():
    data = create_valid_price_data()

    with pytest.raises(
        ValueError,
        match="is not available",
    ):
        add_return_features(
            data,
            price_column="AdjustedClose",
        )


def test_save_and_load_price_data(tmp_path: Path):
    data = create_valid_price_data()
    output_path = tmp_path / "prices.csv"

    save_price_data(data, output_path)
    loaded = load_price_data(output_path)

    pd.testing.assert_frame_equal(
        loaded,
        data,
        check_freq=False,
    )


def test_loading_missing_file_raises_error(tmp_path: Path):
    missing_file = tmp_path / "does_not_exist.csv"

    with pytest.raises(
        FileNotFoundError,
        match="Price data file not found",
    ):
        load_price_data(missing_file)
