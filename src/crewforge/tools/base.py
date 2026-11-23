"""Workspace path-jail security guard and tool provider protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.tools import BaseTool


class WorkspaceSecurityError(Exception):
    """Raised when a tool attempts to access a path outside the target workspace."""


class ToolProvider(Protocol):
    """Protocol implemented by local tool suites and external MCP adapters."""

    def get_tools(self) -> list[BaseTool]:
        """Return LangChain-compatible tools exposed by this provider."""
        ...


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    """Sandboxed filesystem context bound to a single target workspace."""

    workspace_root: Path
    artifact_dir_name: str = ".ai-team"

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_root", self.workspace_root.resolve())

    @property
    def artifact_root(self) -> Path:
        """Absolute path to `<workspace>/.ai-team/`."""
        return (self.workspace_root / self.artifact_dir_name).resolve()

    def ensure_artifact_dir(self) -> Path:
        """Create `<workspace>/.ai-team/` if it does not exist and return its path."""
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        return self.artifact_root

    def resolve_workspace_path(self, relative_path: str) -> Path:
        """Resolve a relative path and ensure it stays inside `workspace_root`."""
        candidate = Path(relative_path)
        if candidate.is_absolute():
            msg = f"Absolute paths are not allowed: '{relative_path}'"
            raise WorkspaceSecurityError(msg)

        resolved = (self.workspace_root / candidate).resolve()
        if not resolved.is_relative_to(self.workspace_root):
            msg = f"Path traversal blocked: '{relative_path}' escapes workspace root."
            raise WorkspaceSecurityError(msg)
        return resolved

    def resolve_artifact_path(self, artifact_name: str) -> Path:
        """Resolve an artifact filename strictly inside `<workspace>/.ai-team/`."""
        candidate = Path(artifact_name)
        if candidate.is_absolute() or len(candidate.parts) != 1:
            msg = (
                f"Artifact name must be a plain filename inside '{self.artifact_dir_name}': "
                f"got '{artifact_name}'"
            )
            raise WorkspaceSecurityError(msg)

        resolved = (self.artifact_root / candidate).resolve()
        if not resolved.is_relative_to(self.artifact_root):
            msg = f"Artifact path traversal blocked: '{artifact_name}'"
            raise WorkspaceSecurityError(msg)
        return resolved