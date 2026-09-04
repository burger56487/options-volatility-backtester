"""PostgreSQL repository tests (skipped unless DATABASE_URL is set)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(
    DATABASE_URL is None,
    reason="DATABASE_URL not set; run with a PostgreSQL service container",
)

from src.storage.importer import import_run_directory
from src.storage.postgres_repository import PostgresRunRepository


@pytest.fixture()
def repository():
    connection = psycopg.connect(DATABASE_URL)
    repository = PostgresRunRepository(connection)
    connection.execute("DELETE FROM runs")
    connection.commit()
    yield repository
    connection.close()


def test_postgres_repository_roundtrip(repository):
    repository.save_run(
        run_id="r1",
        created_at="2026-09-04T00:00:00Z",
        metadata={
            "config_hash": "abc",
            "data": {"option_data_type": "synthetic"},
            "execution": {"mode": "simulated"},
        },
        metrics={"total_pnl": -8038.77},
    )
    record = repository.get_run("r1")
    assert record["run_id"] == "r1"
    assert record["metrics_json"]["total_pnl"] == -8038.77
    assert repository.list_runs()[0]["run_id"] == "r1"
    repository.update_status("r1", "archived")
    assert repository.get_run("r1")["status"] == "archived"


def test_postgres_repository_upsert_is_idempotent(repository):
    repository.save_run(run_id="r2", created_at="2026-09-04T00:00:00Z")
    repository.save_run(
        run_id="r2",
        created_at="2026-09-04T00:00:00Z",
        status="completed",
        metrics={"answer": 42},
    )
    assert len(repository.list_runs()) == 1
    assert repository.get_run("r2")["metrics_json"]["answer"] == 42


def test_postgres_importer_roundtrip(repository, tmp_path: Path):
    (tmp_path / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": "r3",
                "created_at_utc": "2026-09-04T01:00:00Z",
                "data": {"option_data_type": "synthetic"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "research_boundary.json").write_text(
        json.dumps({"boundary": "research-only"}),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"total_pnl": 123.4}),
        encoding="utf-8",
    )
    import_run_directory(repository, tmp_path, "r3")
    record = repository.get_run("r3")
    assert record["status"] == "completed"
    assert record["metrics_json"]["total_pnl"] == 123.4
    assert record["metrics_json"]["research_boundary"]["boundary"] == (
        "research-only"
    )
