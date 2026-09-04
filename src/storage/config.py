"""Environment-driven storage configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorageConfig:
    database_path: Path
    data_root: Path


def load_storage_config() -> StorageConfig:
    """Read configuration from environment variables with sane defaults."""
    database_path = Path(
        os.environ.get(
            "APP_DB_PATH",
            str(Path("outputs") / "app.db"),
        )
    )
    data_root = Path(
        os.environ.get("DATA_ROOT", "outputs")
    )
    return StorageConfig(
        database_path=database_path,
        data_root=data_root,
    )
