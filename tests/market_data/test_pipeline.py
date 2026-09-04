from pathlib import Path

import pandas as pd

from src.market_data.pipeline import run_market_data_pipeline


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_pipeline_cleans_and_reports(tmp_path: Path):
    underlying_path = tmp_path / "underlying.csv"
    option_path = tmp_path / "options.csv"
    _write_csv(
        underlying_path,
        [
            {
                "date": "2026-01-02",
                "symbol": "SPY",
                "open": 100.0,
                "high": 103.0,
                "low": 99.0,
                "close": 102.0,
                "adjusted_close": 102.0,
                "volume": 1_000_000,
            }
        ],
    )
    _write_csv(
        option_path,
        [
            {
                "timestamp": "2026-01-02 16:00:00",
                "underlying_symbol": "SPY",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "option_type": "call",
                "bid": 5.0,
                "ask": 5.2,
                "spot": 102.0,
                "risk_free_rate": 0.04,
                "dividend_yield": 0.01,
                "source": "synthetic_generator",
                "data_type": "synthetic",
            }
        ],
    )

    output = tmp_path / "out"
    summary = run_market_data_pipeline(
        underlying_input_path=underlying_path,
        option_input_path=option_path,
        output_directory=output,
        run_id="test-run-1",
        underlying_source="public_market_data",
    )

    assert (output / "underlying_clean.csv").exists()
    assert (output / "option_quotes_clean.csv").exists()
    assert (output / "data_quality_report.json").exists()
    assert (output / "data_lineage.json").exists()
    assert summary["underlying_records_output"] == 1
    assert summary["option_records_output"] == 1


def test_pipeline_isolates_invalid_records(tmp_path: Path):
    underlying_path = tmp_path / "underlying.csv"
    option_path = tmp_path / "options.csv"
    _write_csv(
        underlying_path,
        [
            {
                "date": "2026-01-02",
                "symbol": "SPY",
                "open": 100.0,
                "high": 99.0,
                "low": 98.0,
                "close": 102.0,
                "adjusted_close": 102.0,
                "volume": 1_000_000,
            }
        ],
    )
    pd.DataFrame(
        columns=[
            "timestamp",
            "underlying_symbol",
            "expiry",
            "strike",
            "option_type",
            "bid",
            "ask",
            "spot",
            "risk_free_rate",
            "dividend_yield",
            "source",
            "data_type",
        ]
    ).to_csv(option_path, index=False)

    import pytest

    with pytest.raises(ValueError, match="Data-quality check failed"):
        run_market_data_pipeline(
            underlying_input_path=underlying_path,
            option_input_path=option_path,
            output_directory=tmp_path / "out2",
            run_id="test-run-2",
            underlying_source="public_market_data",
            fail_on_invalid=True,
        )
