"""Sandboxed workspace and `.ai-team/` artifact filesystem tools."""

from __future__ import annotations

import json
from typing import Any

import yaml
from langchain_core.tools import BaseTool, StructuredTool

from crewforge.tools.base import WorkspaceContext

_IGNORED_DIRS = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache"}
)


def build_filesystem_tools(ctx: WorkspaceContext) -> dict[str, BaseTool]:
    """Build workspace-scoped file and artifact tools."""

    def read_artifact(artifact_name: str) -> str:
        """Read a structured artifact file from `.ai-team/`."""
        path = ctx.resolve_artifact_path(artifact_name)
        if not path.is_file():
            return f"Artifact not found: {artifact_name}"
        return path.read_text(encoding="utf-8")

    def write_artifact(
        artifact_name: str,
        content: str | dict[str, Any] | list[Any],
    ) -> str:
        """Write or overwrite a structured artifact file inside `.ai-team/`."""
        ctx.ensure_artifact_dir()
        path = ctx.resolve_artifact_path(artifact_name)
        if isinstance(content, str):
            text = content
        elif artifact_name.endswith((".yaml", ".yml")):
            text = yaml.safe_dump(content, sort_keys=False)
        else:
            text = json.dumps(content, indent=2)
        path.write_text(text, encoding="utf-8")
        rel = path.relative_to(ctx.workspace_root).as_posix()
        return f"Wrote artifact '{artifact_name}' ({len(text)} chars) to {rel}."

    def list_workspace_files(subdir: str = ".", max_entries: int = 100) -> str:
        """List files inside the target workspace, skipping common build/cache directories."""
        root = ctx.resolve_workspace_path(subdir)
        if not root.is_dir():
            return f"Directory not found: {subdir}"

        results: list[str] = []
        for path in sorted(root.rglob("*")):
            if any(part in _IGNORED_DIRS for part in path.parts):
                continue
            if path.is_file():
                results.append(path.relative_to(ctx.workspace_root).as_posix())
                if len(results) >= max_entries:
                    results.append(f"... (truncated at {max_entries} entries)")
                    break
        return "\n".join(results) if results else "(no files found)"

    def read_workspace_file(relative_path: str) -> str:
        """Read a UTF-8 text file inside the target workspace."""
        path = ctx.resolve_workspace_path(relative_path)
        if not path.is_file():
            return f"File not found: {relative_path}"
        return path.read_text(encoding="utf-8")

    def write_workspace_file(relative_path: str, content: str) -> str:
        """Create or overwrite a UTF-8 text file inside the target workspace."""
        path = ctx.resolve_workspace_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        rel = path.relative_to(ctx.workspace_root).as_posix()
        return f"Wrote {len(content)} chars to {rel}."

    tools: list[BaseTool] = [
        StructuredTool.from_function(func=read_artifact, name="read_artifact"),
        StructuredTool.from_function(func=write_artifact, name="write_artifact"),
        StructuredTool.from_function(func=list_workspace_files, name="list_workspace_files"),
        StructuredTool.from_function(func=read_workspace_file, name="read_workspace_file"),
        StructuredTool.from_function(func=write_workspace_file, name="write_workspace_file"),
    ]
    return {t.name: t for t in tools}