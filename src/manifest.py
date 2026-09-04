"""Output-file manifest generation with hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(output_directory: str | Path) -> Path:
    """Write a manifest of all files under the output directory."""
    output_path = Path(output_directory)
    files = []
    for path in sorted(output_path.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(output_path)),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"files": files},
            file,
            ensure_ascii=False,
            indent=2,
        )
    return manifest_path
