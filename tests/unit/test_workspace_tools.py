"""Unit tests for workspace-sandboxed tools, path jail, and MCP adapter seam."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.tools import StructuredTool

from crewforge.tools import (
    MCPServerConfig,
    MCPToolAdapter,
    WorkspaceContext,
    WorkspaceSecurityError,
    build_tool_registry,
    select_tools_for_agent,
)


def test_path_jail_blocks_traversal_and_absolute_paths(tmp_path: Path) -> None:
    ctx = WorkspaceContext(workspace_root=tmp_path)

    with pytest.raises(WorkspaceSecurityError, match="Path traversal"):
        ctx.resolve_workspace_path("../outside.txt")

    with pytest.raises(WorkspaceSecurityError, match="Absolute paths"):
        ctx.resolve_workspace_path(str(tmp_path / "file.txt"))

    with pytest.raises(WorkspaceSecurityError, match="plain filename"):
        ctx.resolve_artifact_path("nested/01_prd.md")


def test_artifact_and_workspace_file_tools(tmp_path: Path) -> None:
    ctx = WorkspaceContext(workspace_root=tmp_path)
    registry = build_tool_registry(ctx)

    write_msg = registry["write_artifact"].invoke(
        {"artifact_name": "01_prd.md", "content": "# PRD\nBuild health endpoint."}
    )
    assert "01_prd.md" in str(write_msg)

    read_content = registry["read_artifact"].invoke({"artifact_name": "01_prd.md"})
    assert "Build health endpoint." in str(read_content)

    registry["write_workspace_file"].invoke(
        {"relative_path": "src/app.py", "content": "print('hello')\n"}
    )
    listed = str(registry["list_workspace_files"].invoke({}))
    assert "src/app.py" in listed
    assert str(registry["read_workspace_file"].invoke({"relative_path": "src/app.py"})) == (
        "print('hello')\n"
    )


def test_shell_tool_enforces_allowlist(tmp_path: Path) -> None:
    ctx = WorkspaceContext(workspace_root=tmp_path)
    registry = build_tool_registry(ctx, allowed_shell_prefixes=("echo ",))

    ok_output = str(registry["run_shell_command"].invoke({"command": "echo crewforge"}))
    assert "exit_code=0" in ok_output
    assert "crewforge" in ok_output

    with pytest.raises(WorkspaceSecurityError, match="allowlist"):
        registry["run_shell_command"].invoke({"command": "rm -rf /"})


def test_mcp_adapter_and_least_privilege_selection(tmp_path: Path) -> None:
    def ping_mcp() -> str:
        """Sample MCP-provided tool."""
        return "pong"

    mcp_tool = StructuredTool.from_function(func=ping_mcp, name="mcp_ping")
    adapter = MCPToolAdapter(server=MCPServerConfig(name="demo"), tools=[mcp_tool])

    ctx = WorkspaceContext(workspace_root=tmp_path)
    registry = build_tool_registry(ctx, extra_providers=[adapter])

    selected = select_tools_for_agent(registry, ["read_artifact", "mcp_ping"])
    assert [t.name for t in selected] == ["read_artifact", "mcp_ping"]
    assert selected[1].invoke({}) == "pong"