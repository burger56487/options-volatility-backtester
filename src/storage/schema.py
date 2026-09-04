"""Lightweight schema and ordered migrations."""

from __future__ import annotations

import sqlite3


_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    config_hash TEXT,
    option_data_type TEXT,
    execution_type TEXT,
    evaluation_mode TEXT,
    metadata_json TEXT,
    metrics_json TEXT
);
"""


MIGRATIONS = [_SCHEMA_V1]


def apply_migrations(connection: sqlite3.Connection) -> None:
    """Apply all unapplied migrations, tracked by schema_version."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        )
        """
    )
    current = connection.execute(
        "SELECT MAX(version) FROM schema_version"
    ).fetchone()[0]
    current = int(current) if current is not None else 0
    for index in range(current, len(MIGRATIONS)):
        connection.executescript(MIGRATIONS[index])
        connection.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (index + 1,),
        )
    connection.commit()
