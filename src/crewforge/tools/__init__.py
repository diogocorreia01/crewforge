"""Workspace-sandboxed tool registry and MCP adapter seam."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from langchain_core.tools import BaseTool

from crewforge.tools.base import ToolProvider, WorkspaceContext, WorkspaceSecurityError
from crewforge.tools.filesystem import build_filesystem_tools
from crewforge.tools.git import build_git_tools
from crewforge.tools.mcp_adapter import MCPServerConfig, MCPToolAdapter
from crewforge.tools.shell import build_shell_tools


def build_tool_registry(
    ctx: WorkspaceContext,
    *,
    extra_providers: Sequence[ToolProvider] = (),
    shell_timeout_seconds: int = 60,
    allowed_shell_prefixes: tuple[str, ...] | None = None,
) -> dict[str, BaseTool]:
    """Assemble the full workspace-scoped tool registry (local + optional MCP tools)."""
    registry: dict[str, BaseTool] = {}
    registry.update(build_filesystem_tools(ctx))
    registry.update(
        build_shell_tools(
            ctx,
            timeout_seconds=shell_timeout_seconds,
            allowed_prefixes=allowed_shell_prefixes,
        )
    )
    registry.update(build_git_tools(ctx))
    for provider in extra_providers:
        for tool in provider.get_tools():
            registry[tool.name] = tool
    return registry


def select_tools_for_agent(
    registry: Mapping[str, BaseTool],
    allowed_tools: Sequence[str],
) -> list[BaseTool]:
    """Return only the tools explicitly granted to an agent in `agents.yaml`."""
    selected: list[BaseTool] = []
    for name in allowed_tools:
        if name not in registry:
            msg = f"Agent requested unknown tool '{name}' (available: {sorted(registry)})"
            raise KeyError(msg)
        selected.append(registry[name])
    return selected


__all__ = [
    "MCPServerConfig",
    "MCPToolAdapter",
    "ToolProvider",
    "WorkspaceContext",
    "WorkspaceSecurityError",
    "build_tool_registry",
    "select_tools_for_agent",
]