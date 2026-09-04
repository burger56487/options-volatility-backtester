"""Institutional lock preventing casual repeated test-set evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def check_test_evaluation_allowed(
    lock_file: str | Path,
    allow_repeat: bool = False,
) -> None:
    path = Path(lock_file)
    if path.exists() and not allow_repeat:
        with path.open("r", encoding="utf-8") as file:
            prior = json.load(file)
        raise RuntimeError(
            "Test set already evaluated at "
            f"{prior['evaluated_at_utc']}. Set allow_repeat=True with "
            "a documented reason to re-evaluate."
        )


def record_test_evaluation(
    lock_file: str | Path,
    run_id: str,
    selected_parameters: dict,
    git_commit: str | None,
    reason: str | None = None,
) -> None:
    path = Path(lock_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_parameters": selected_parameters,
        "git_commit": git_commit,
        "reason": reason,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
