"""Run the market-data quality pipeline on the sample dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.market_data.pipeline import run_market_data_pipeline
from src.run_context import initialise_run


def main() -> None:
    config = load_config("configs/default.yaml")
    context = initialise_run(
        config=config,
        config_path="configs/default.yaml",
        command="python scripts/run_data_pipeline.py",
    )

    data_quality = config.get("data_quality", {})
    output = context.output_directory / "market_data"

    summary = run_market_data_pipeline(
        underlying_input_path=Path(
            "data/sample/underlying_sample.csv"
        ),
        option_input_path=Path(
            "data/sample/option_quotes_sample.csv"
        ),
        output_directory=output,
        run_id=context.run_id,
        underlying_source=config["data"]["underlying"]["source"],
        fail_on_invalid=data_quality.get("fail_on_invalid", True),
        max_relative_spread=data_quality.get(
            "max_option_relative_spread",
            0.50,
        ),
        arbitrage_tolerance=data_quality.get(
            "arbitrage_tolerance",
            1e-8,
        ),
    )

    print(f"run_id: {context.run_id}")
    print(f"output_directory: {output}")
    print(
        f"underlying records: {summary['underlying_records_output']}, "
        f"option records: {summary['option_records_output']}"
    )
    print(f"quality report: {summary['quality_report']}")


if __name__ == "__main__":
    main()
