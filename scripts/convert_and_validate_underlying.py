"""Convert the legacy SPY CSV to the unified schema and run the pipeline.

The legacy file stores adjusted close values in the Close column and carries
Open/High/Low/Volume, so it can be mapped to the unified UnderlyingBar schema
without losing information.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import load_config
from src.market_data.pipeline import run_market_data_pipeline
from src.run_context import initialise_run


def main() -> None:
    legacy_path = Path("data/raw/spy_daily_adjusted.csv")
    raw_underlying_dir = Path("data/raw/underlying")
    raw_underlying_dir.mkdir(parents=True, exist_ok=True)
    unified_path = raw_underlying_dir / "underlying.csv"

    legacy = pd.read_csv(legacy_path)
    legacy["Date"] = pd.to_datetime(legacy["Date"]).dt.date
    unified = pd.DataFrame(
        {
            "date": legacy["Date"],
            "symbol": "SPY",
            "open": legacy["Open"],
            "high": legacy["High"],
            "low": legacy["Low"],
            "close": legacy["Close"],
            "adjusted_close": legacy["Close"],
            "volume": legacy["Volume"],
        }
    )
    unified.to_csv(unified_path, index=False)

    config = load_config("configs/default.yaml")
    context = initialise_run(
        config=config,
        config_path="configs/default.yaml",
        command="python scripts/convert_and_validate_underlying.py",
    )
    data_quality = config.get("data_quality", {})
    output = context.output_directory / "market_data"
    summary = run_market_data_pipeline(
        underlying_input_path=unified_path,
        option_input_path=Path(
            "data/sample/option_quotes_sample.csv"
        ),
        output_directory=output,
        run_id=context.run_id,
        underlying_source=config["data"]["underlying"]["source"],
        fail_on_invalid=data_quality.get("fail_on_invalid", True),
    )
    print(f"run_id: {context.run_id}")
    print(
        f"underlying bars written: {summary['underlying_records_output']}"
    )
    print(f"clean file: {summary['underlying_output']}")


if __name__ == "__main__":
    main()
