"""Run-result repositories (protocol, SQLite, in-memory)."""

from __future__ import annotations

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .schema import apply_migrations


class RunRepository(ABC):
    @abstractmethod
    def save_run(
        self,
        run_id: str,
        created_at: str,
        status: str = "completed",
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        ...

    @abstractmethod
    def get_run(self, run_id: str) -> dict[str, Any] | None:
        ...

    @abstractmethod
    def list_runs(self) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def update_status(self, run_id: str, status: str) -> None:
        ...


class SqliteRunRepository(RunRepository):
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        apply_migrations(connection)

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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status = excluded.status,
                metadata_json = excluded.metadata_json,
                metrics_json = excluded.metrics_json
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
            "SELECT * FROM runs WHERE run_id = ?",
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
            "UPDATE runs SET status = ? WHERE run_id = ?",
            (status, run_id),
        )
        self.connection.commit()


class InMemoryRunRepository(RunRepository):
    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}

    def save_run(self, run_id, created_at, status="completed",
                 metadata=None, metrics=None):
        self._runs[run_id] = {
            "run_id": run_id,
            "created_at": created_at,
            "status": status,
            "metadata_json": metadata or {},
            "metrics_json": metrics or {},
        }

    def get_run(self, run_id):
        return self._runs.get(run_id)

    def list_runs(self):
        return [
            {
                "run_id": run["run_id"],
                "created_at": run["created_at"],
                "status": run["status"],
            }
            for run in sorted(
                self._runs.values(),
                key=lambda item: item["created_at"],
                reverse=True,
            )
        ]

    def update_status(self, run_id, status):
        if run_id in self._runs:
            self._runs[run_id]["status"] = status


def connect_run_repository() -> RunRepository:
    """Build the repository from ``DATABASE_URL`` or ``APP_DB_PATH``.

    When ``DATABASE_URL`` is set a PostgreSQL repository is used (psycopg
    required); otherwise the default is a SQLite database under ``outputs/``.
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        import psycopg

        from .postgres_repository import PostgresRunRepository

        return PostgresRunRepository(psycopg.connect(database_url))
    database_path = Path(
        os.environ.get(
            "APP_DB_PATH",
            str(Path("outputs") / "app.db"),
        )
    )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        str(database_path),
        check_same_thread=False,
    )
    return SqliteRunRepository(connection)
