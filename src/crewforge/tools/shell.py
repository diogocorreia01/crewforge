"""Workspace-scoped shell execution tool with timeout and optional allowlist."""

from __future__ import annotations

import subprocess

from langchain_core.tools import BaseTool, StructuredTool

from crewforge.tools.base import WorkspaceContext, WorkspaceSecurityError

_MAX_OUTPUT_CHARS = 8000


def build_shell_tools(
    ctx: WorkspaceContext,
    *,
    timeout_seconds: int = 60,
    allowed_prefixes: tuple[str, ...] | None = None,
) -> dict[str, BaseTool]:
    """Build a scoped shell runner bound to `ctx.workspace_root`."""

    def run_shell_command(command: str) -> str:
        """Run a shell command inside the target workspace (for tests, linters, builds)."""
        stripped = command.strip()
        if not stripped:
            return "Error: command cannot be empty."

        if allowed_prefixes is not None and not any(
            stripped.startswith(prefix) for prefix in allowed_prefixes
        ):
            msg = f"Command not permitted by allowlist: '{stripped}'"
            raise WorkspaceSecurityError(msg)

        try:
            proc = subprocess.run(
                stripped,
                shell=True,
                cwd=ctx.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout_seconds}s: {stripped}"

        combined = f"exit_code={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        if len(combined) > _MAX_OUTPUT_CHARS:
            return combined[:_MAX_OUTPUT_CHARS] + "\n... (output truncated)"
        return combined

    tool = StructuredTool.from_function(func=run_shell_command, name="run_shell_command")
    return {tool.name: tool}