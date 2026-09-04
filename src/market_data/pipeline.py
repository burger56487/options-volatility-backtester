"""End-to-end market-data quality pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .cleaning import (
    deduplicate_option_quotes,
    deduplicate_underlying_bars,
    find_duplicate_option_quotes,
    find_duplicate_underlying_bars,
    option_quote_key,
    underlying_key,
)
from .lineage import create_data_lineage, save_data_lineage
from .loaders import load_option_quotes_csv, load_underlying_csv
from .quality_report import save_quality_report, summarise_validation_results
from .validators import (
    ValidationResult,
    validate_option_quote,
    validate_underlying_bar,
)


def records_to_dataframe(records: list) -> pd.DataFrame:
    return pd.DataFrame([record.to_dict() for record in records])


def validation_issues_to_dataframe(
    records: list,
    results: list[ValidationResult],
    key_builder: Callable,
) -> pd.DataFrame:
    rows = []
    for record, result in zip(records, results):
        for issue in result.issues:
            rows.append(
                {
                    "record_key": str(key_builder(record)),
                    "code": issue.code,
                    "severity": issue.severity.value,
                    "field": issue.field,
                    "value": issue.value,
                    "message": issue.message,
                }
            )
    return pd.DataFrame(rows)


def run_market_data_pipeline(
    underlying_input_path: str | Path,
    option_input_path: str | Path,
    output_directory: str | Path,
    run_id: str,
    underlying_source: str,
    fail_on_invalid: bool = True,
    max_relative_spread: float = 0.50,
    arbitrage_tolerance: float = 1e-8,
) -> dict:
    """Validate, clean and version one market-data snapshot."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    underlying_bars = load_underlying_csv(
        path=underlying_input_path,
        source=underlying_source,
    )
    option_quotes = load_option_quotes_csv(path=option_input_path)

    underlying_duplicates = find_duplicate_underlying_bars(
        underlying_bars
    )
    option_duplicates = find_duplicate_option_quotes(option_quotes)

    underlying_results = [
        validate_underlying_bar(bar) for bar in underlying_bars
    ]
    option_results = [
        validate_option_quote(
            quote,
            max_relative_spread=max_relative_spread,
            arbitrage_tolerance=arbitrage_tolerance,
        )
        for quote in option_quotes
    ]

    quality_summaries = [
        summarise_validation_results(
            dataset_name="underlying",
            validation_results=underlying_results,
            duplicate_records=sum(
                count - 1
                for count in underlying_duplicates.values()
            ),
        ),
        summarise_validation_results(
            dataset_name="options",
            validation_results=option_results,
            duplicate_records=sum(
                count - 1
                for count in option_duplicates.values()
            ),
        ),
    ]
    quality_report_path = output_path / "data_quality_report.json"
    save_quality_report(
        summaries=quality_summaries,
        output_path=quality_report_path,
    )

    underlying_issues = validation_issues_to_dataframe(
        underlying_bars,
        underlying_results,
        underlying_key,
    )
    option_issues = validation_issues_to_dataframe(
        option_quotes,
        option_results,
        option_quote_key,
    )
    underlying_issues_path = (
        output_path / "underlying_validation_issues.csv"
    )
    option_issues_path = output_path / "option_validation_issues.csv"
    underlying_issues.to_csv(
        underlying_issues_path,
        index=False,
    )
    option_issues.to_csv(option_issues_path, index=False)

    cleaned_underlying = [
        bar
        for bar, result in zip(underlying_bars, underlying_results)
        if result.valid
    ]
    cleaned_options = [
        quote
        for quote, result in zip(option_quotes, option_results)
        if result.valid
    ]
    cleaned_underlying = deduplicate_underlying_bars(
        cleaned_underlying
    )
    cleaned_options = deduplicate_option_quotes(cleaned_options)

    invalid_count = sum(
        not result.valid for result in underlying_results
    ) + sum(not result.valid for result in option_results)

    if fail_on_invalid and invalid_count > 0:
        raise ValueError(
            f"Data-quality check failed with {invalid_count} "
            f"invalid records. Details: {quality_report_path}"
        )

    underlying_output_path = output_path / "underlying_clean.csv"
    option_output_path = output_path / "option_quotes_clean.csv"
    records_to_dataframe(cleaned_underlying).to_csv(
        underlying_output_path,
        index=False,
    )
    records_to_dataframe(cleaned_options).to_csv(
        option_output_path,
        index=False,
    )

    lineage = create_data_lineage(
        input_files=[underlying_input_path, option_input_path],
        output_files=[
            underlying_output_path,
            option_output_path,
            quality_report_path,
            underlying_issues_path,
            option_issues_path,
        ],
        transformations=[
            "schema_validation",
            "business_rule_validation",
            "european_no_arbitrage_validation",
            "invalid_record_filtering",
            "duplicate_removal",
            "sorting",
        ],
        run_id=run_id,
    )
    lineage_path = save_data_lineage(
        lineage=lineage,
        output_path=output_path / "data_lineage.json",
    )

    summary = {
        "run_id": run_id,
        "underlying_records_output": len(cleaned_underlying),
        "option_records_output": len(cleaned_options),
        "quality_report": str(quality_report_path),
        "data_lineage": str(lineage_path),
        "underlying_output": str(underlying_output_path),
        "option_output": str(option_output_path),
    }
    with (output_path / "pipeline_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary
