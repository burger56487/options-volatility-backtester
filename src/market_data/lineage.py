"""Data lineage and file-hash tracking."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_data_lineage(
    input_files: list[str | Path],
    output_files: list[str | Path],
    transformations: list[str],
    run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in input_files
        ],
        "outputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in output_files
            if Path(path).exists()
        ],
        "transformations": transformations,
    }


def save_data_lineage(
    lineage: dict[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(lineage, file, ensure_ascii=False, indent=2)
    return path
