"""Traceable run metadata for reproducible research results."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_git_is_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def get_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def create_config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def create_run_metadata(
    config: dict[str, Any],
    command: str | None = None,
) -> dict[str, Any]:
    """Build the full metadata record for one research run."""
    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    ) + "_" + uuid.uuid4().hex[:8]

    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": config["project"]["name"],
            "version": config["project"]["version"],
            "research_purpose": config["project"]["research_purpose"],
        },
        "data": {
            "underlying_data_type": (
                config["data"]["underlying"]["data_type"]
            ),
            "underlying_source": (
                config["data"]["underlying"]["source"]
            ),
            "option_data_type": (
                config["data"]["options"]["data_type"]
            ),
            "option_source": (
                config["data"]["options"]["source"]
            ),
        },
        "execution": {
            "mode": config["execution"]["mode"],
            "commission_enabled": (
                config["execution"]["commission_enabled"]
            ),
            "slippage_enabled": (
                config["execution"]["slippage_enabled"]
            ),
            "market_impact_enabled": (
                config["execution"]["market_impact_enabled"]
            ),
        },
        "research": {
            "evaluation_mode": (
                config["research"]["evaluation_mode"]
            ),
            "signal_delay_days": (
                config["research"]["signal_delay_days"]
            ),
            "prevent_lookahead": (
                config["research"]["prevent_lookahead"]
            ),
            "random_seed": config["research"]["random_seed"],
        },
        "reproducibility": {
            "git_commit": get_git_commit(),
            "git_worktree_dirty": get_git_is_dirty(),
            "config_hash": create_config_hash(config),
            "command": command,
        },
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": {
                "numpy": get_package_version("numpy"),
                "pandas": get_package_version("pandas"),
                "scipy": get_package_version("scipy"),
                "matplotlib": get_package_version("matplotlib"),
                "pyyaml": get_package_version("PyYAML"),
            },
        },
        "disclaimer": (
            "This result is based on research-style simulated execution. "
            "If option data type is synthetic, results do not represent "
            "real historical options-market performance."
        ),
    }


def save_run_metadata(
    metadata: dict[str, Any],
    output_directory: str | Path,
) -> Path:
    """Save run metadata to ``run_metadata.json`` under the directory."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    return metadata_path
