"""Run the API server (uvicorn app factory entry)."""

from __future__ import annotations

import sqlite3

from src.api.app import create_app
from src.storage.repository import SqliteRunRepository


def app():
    connection = sqlite3.connect("outputs/app.db", check_same_thread=False)
    repository = SqliteRunRepository(connection)
    return create_app(repository=repository)


application = app()
