"""LangGraph checkpointer factory (SQLite for persistence, Memory for tests)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from crewforge.config.schemas import CheckpointerBackend, CheckpointerConfig
from crewforge.tools.base import WorkspaceContext


@contextmanager
def open_checkpointer(
    config: CheckpointerConfig,
    workspace_ctx: WorkspaceContext,
) -> Iterator[BaseCheckpointSaver[Any]]:
    """Yield a configured LangGraph checkpointer and close resources on exit."""
    if config.backend == CheckpointerBackend.MEMORY:
        yield MemorySaver()
        return

    workspace_ctx.ensure_artifact_dir()
    db_path = workspace_ctx.resolve_artifact_path(config.sqlite_filename)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()