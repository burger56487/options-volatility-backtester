"""Run the API server (uvicorn app factory entry)."""

from __future__ import annotations

from src.api.app import create_app
from src.storage.repository import connect_run_repository


def app():
    return create_app(repository=connect_run_repository())


application = app()
