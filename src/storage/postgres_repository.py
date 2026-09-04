"""PostgreSQL run-result repository with ordered migrations and indexes."""

from __future__ import annotations

import json
from typing import Any

from .repository import RunRepository


_MIGRATIONS = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL
    )
    """,
    """
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
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_runs_created_at
    ON runs (created_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_runs_status
    ON runs (status)
    """,
)


def apply_migrations_postgres(connection) -> None:
    """Apply unapplied migrations, tracked in ``schema_version``."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        )
        """
    )
    row = connection.execute(
        "SELECT MAX(version) FROM schema_version"
    ).fetchone()
    current = int(row[0]) if row is not None and row[0] is not None else 0
    for index in range(current, len(_MIGRATIONS)):
        connection.execute(_MIGRATIONS[index])
        connection.execute(
            "INSERT INTO schema_version (version) VALUES (%s)",
            (index + 1,),
        )
    connection.commit()


class PostgresRunRepository(RunRepository):
    """Postgres-backed implementation of :class:`RunRepository`."""

    def __init__(self, connection) -> None:
        self.connection = connection
        apply_migrations_postgres(connection)

    def save_run(
        self,
        run_id: str,
        created_at: str,
        status: str = "completed",
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata or {}
        metrics = metrics or {}
        self.connection.execute(
            """
            INSERT INTO runs (
                run_id, created_at, status, config_hash,
                option_data_type, execution_type, evaluation_mode,
                metadata_json, metrics_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                status = EXCLUDED.status,
                metadata_json = EXCLUDED.metadata_json,
                metrics_json = EXCLUDED.metrics_json
            """,
            (
                run_id,
                created_at,
                status,
                (metadata or {}).get("config_hash"),
                (metadata.get("data") or {}).get("option_data_type"),
                (metadata.get("execution") or {}).get("mode"),
                (metadata.get("research") or {}).get("evaluation_mode"),
                json.dumps(metadata, default=str),
                json.dumps(metrics, default=str),
            ),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        cursor = self.connection.execute(
            "SELECT * FROM runs WHERE run_id = %s",
            (run_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [item[0] for item in cursor.description]
        record = dict(zip(columns, row))
        record["metadata_json"] = json.loads(record["metadata_json"] or "{}")
        record["metrics_json"] = json.loads(record["metrics_json"] or "{}")
        return record

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT run_id, created_at, status FROM runs "
            "ORDER BY created_at DESC"
        ).fetchall()
        return [
            {
                "run_id": row[0],
                "created_at": row[1],
                "status": row[2],
            }
            for row in rows
        ]

    def update_status(self, run_id: str, status: str) -> None:
        self.connection.execute(
            "UPDATE runs SET status = %s WHERE run_id = %s",
            (status, run_id),
        )
        self.connection.commit()
