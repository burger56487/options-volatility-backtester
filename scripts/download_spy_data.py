from pathlib import Path

from src.market_data.underlying_data import (
    add_return_features,
    clean_price_data,
    download_price_data,
    save_price_data,
    validate_price_data,
)


def main() -> None:
    ticker = "SPY"
    start = "2021-01-01"
    end = "2026-01-01"

    print("=" * 64)
    print("UNDERLYING PRICE DATA DOWNLOAD AND VALIDATION")
    print("=" * 64)
    print()

    print(f"Ticker: {ticker}")
    print(f"Date range: {start} to {end}")
    print()

    data = download_price_data(
        ticker=ticker,
        start=start,
        end=end,
    )

    report = validate_price_data(data)

    print("Data quality report")
    print("-" * 64)
    print(f"Observations: {report.observations}")
    print(f"Start date: {report.start_date}")
    print(f"End date: {report.end_date}")
    print(f"Duplicate dates: {report.duplicate_dates}")
    print(f"Missing values: {report.missing_values}")
    print(f"Non-positive prices: {report.non_positive_prices}")
    print(f"Invalid OHLC rows: {report.invalid_high_low_rows}")
    print(f"Negative volume rows: {report.negative_volume_rows}")
    print()

    clean_data = clean_price_data(data)
    featured_data = add_return_features(clean_data)

    output_path = Path("data/raw/spy_daily_adjusted.csv")
    save_price_data(featured_data, output_path)

    print(f"Saved data to: {output_path}")
    print()
    print("Last five observations")
    print("-" * 64)
    print(featured_data.tail().round(6).to_string())


if __name__ == "__main__":
    main()
