"""Machine-readable data-quality reporting."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .validators import ValidationResult


@dataclass
class DatasetQualitySummary:
    dataset_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    records_with_warnings: int
    duplicate_records: int
    retention_rate: float
    error_counts: dict[str, int]
    warning_counts: dict[str, int]


def summarise_validation_results(
    dataset_name: str,
    validation_results: Iterable[ValidationResult],
    duplicate_records: int,
) -> DatasetQualitySummary:
    results = list(validation_results)
    valid_records = sum(result.valid for result in results)
    records_with_warnings = sum(
        bool(result.warnings) for result in results
    )
    error_counts = Counter(
        issue.code
        for result in results
        for issue in result.errors
    )
    warning_counts = Counter(
        issue.code
        for result in results
        for issue in result.warnings
    )
    total_records = len(results)
    invalid_records = total_records - valid_records
    retention_rate = (
        valid_records / total_records if total_records > 0 else 0.0
    )
    return DatasetQualitySummary(
        dataset_name=dataset_name,
        total_records=total_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        records_with_warnings=records_with_warnings,
        duplicate_records=duplicate_records,
        retention_rate=retention_rate,
        error_counts=dict(error_counts),
        warning_counts=dict(warning_counts),
    )


def save_quality_report(
    summaries: list[DatasetQualitySummary],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "datasets": [asdict(summary) for summary in summaries]
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return path
