"""Adapter seam for mounting external Model Context Protocol (MCP) tool servers."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


class MCPServerConfig(BaseModel):
    """Connection metadata for an external MCP server."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    transport: str = Field(default="stdio", description="e.g. 'stdio' or 'sse'.")
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None


class MCPToolAdapter:
    """Placeholder ToolProvider seam for plugging MCP server tools into CrewForge.

    Implement `get_tools()` with `langchain-mcp-adapters` when connecting live MCP servers.
    """

    def __init__(self, server: MCPServerConfig, tools: list[BaseTool] | None = None) -> None:
        self.server = server
        self._tools: list[BaseTool] = list(tools) if tools is not None else []

    def get_tools(self) -> list[BaseTool]:
        """Return adapted LangChain tools exposed by the MCP server."""
        return list(self._tools)