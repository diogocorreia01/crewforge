"""Workspace-scoped read-only Git inspection tools."""

from __future__ import annotations

import subprocess

from langchain_core.tools import BaseTool, StructuredTool

from crewforge.tools.base import WorkspaceContext


def _run_git(ctx: WorkspaceContext, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ctx.workspace_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        return f"git error (exit {proc.returncode}): {proc.stderr.strip()}"
    return proc.stdout.strip() or "(clean)"


def build_git_tools(ctx: WorkspaceContext) -> dict[str, BaseTool]:
    """Build safe git inspection tools scoped to `ctx.workspace_root`."""

    def git_status() -> str:
        """Return short git status for the target workspace."""
        return _run_git(ctx, ["status", "--short"])

    def git_diff(relative_path: str = "") -> str:
        """Return unstaged git diff for the workspace or a specific relative file path."""
        args = ["diff"]
        if relative_path.strip():
            resolved = ctx.resolve_workspace_path(relative_path)
            args.extend(["--", resolved.relative_to(ctx.workspace_root).as_posix()])
        return _run_git(ctx, args)

    tools: list[BaseTool] = [
        StructuredTool.from_function(func=git_status, name="git_status"),
        StructuredTool.from_function(func=git_diff, name="git_diff"),
    ]
    return {t.name: t for t in tools}