"""Background run manager backed by a RunRepository."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Callable

from src.storage.repository import RunRepository


class JobManager:
    """Submit callables as background jobs with status persisted to repo."""

    def __init__(self, repository: RunRepository) -> None:
        self.repository = repository
        self._threads: dict[str, threading.Thread] = {}

    def submit(
        self,
        run_id: str,
        task: Callable[[], dict],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.repository.save_run(
            run_id=run_id,
            created_at=now,
            status="running",
        )

        def worker() -> None:
            try:
                metrics = task()
                self.repository.save_run(
                    run_id=run_id,
                    created_at=now,
                    status="completed",
                    metrics=metrics,
                )
            except Exception as exc:  # noqa: BLE001
                self.repository.save_run(
                    run_id=run_id,
                    created_at=now,
                    status="failed",
                    metrics={"error": str(exc)},
                )

        thread = threading.Thread(target=worker, daemon=True)
        self._threads[run_id] = thread
        thread.start()
