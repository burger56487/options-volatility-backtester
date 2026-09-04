import json
import sqlite3
from pathlib import Path

from src.storage.importer import import_run_directory
from src.storage.repository import (
    InMemoryRunRepository,
    SqliteRunRepository,
)
from src.storage.schema import apply_migrations


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_sqlite_repository_roundtrip(tmp_path):
    connection = sqlite3.connect(tmp_path / "app.db")
    repository = SqliteRunRepository(connection)
    repository.save_run(
        run_id="r1",
        created_at="2026-09-04T00:00:00Z",
        metadata={"data": {"option_data_type": "synthetic"}},
        metrics={"total_pnl": -8038.77},
    )
    record = repository.get_run("r1")
    assert record["run_id"] == "r1"
    assert record["metrics_json"]["total_pnl"] == -8038.77
    assert repository.list_runs()[0]["run_id"] == "r1"
    repository.update_status("r1", "archived")
    assert repository.get_run("r1")["status"] == "archived"


def test_migrations_are_idempotent(tmp_path):
    connection = sqlite3.connect(tmp_path / "mig.db")
    apply_migrations(connection)
    apply_migrations(connection)
    version = connection.execute(
        "SELECT MAX(version) FROM schema_version"
    ).fetchone()[0]
    assert version == 1


def test_import_run_directory(tmp_path):
    _write(
        tmp_path,
        "run_metadata.json",
        {
            "run_id": "run-1",
            "created_at_utc": "2026-09-04T00:00:00Z",
            "data": {"option_data_type": "synthetic"},
        },
    )
    _write(
        tmp_path,
        "research_boundary.json",
        {"option_data": "synthetic", "execution": "simulated"},
    )
    _write(
        tmp_path,
        "summary.json",
        {"total_pnl": -8038.77, "number_of_trades": 17},
    )
    repository = InMemoryRunRepository()
    import_run_directory(repository, tmp_path, "run-1")
    record = repository.get_run("run-1")
    assert record["metrics_json"]["total_pnl"] == -8038.77
    assert (
        record["metrics_json"]["research_boundary"]["execution"]
        == "simulated"
    )
