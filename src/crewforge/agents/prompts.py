"""Prompt assembly with targeted `.ai-team/` artifact injection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from crewforge.config.schemas import AgentConfig
from crewforge.state.project_state import ProjectState
from crewforge.tools.base import WorkspaceContext


def load_upstream_artifacts(
    ctx: WorkspaceContext,
    artifact_names: Sequence[str],
) -> dict[str, str]:
    """Read existing upstream artifacts listed in an agent's `reads_artifacts` contract."""
    loaded: dict[str, str] = {}
    for name in artifact_names:
        path = ctx.resolve_artifact_path(name)
        if path.is_file():
            loaded[name] = path.read_text(encoding="utf-8")
    return loaded


def build_agent_messages(
    *,
    agent_cfg: AgentConfig,
    state: ProjectState,
    upstream_artifacts: Mapping[str, str],
) -> list[BaseMessage]:
    """Assemble compact system + human messages without cross-agent chat history."""
    sections: list[str] = [
        f"## Feature Request\n{state.get('feature_request', '').strip()}",
    ]

    hitl_feedback = state.get("hitl_feedback", "").strip()
    if hitl_feedback:
        sections.append(f"## Human Reviewer Feedback\n{hitl_feedback}")

    qa_retry_count = state.get("qa_retry_count", 0)
    if qa_retry_count > 0:
        sections.append(
            f"## QA Fix Cycle\nThis is fix attempt #{qa_retry_count}. "
            "Prioritize resolving all issues reported in `06_qa_report.json`."
        )

    if upstream_artifacts:
        artifact_blocks = [
            f"### Artifact: `{name}`\n```text\n{content}\n```"
            for name, content in upstream_artifacts.items()
        ]
        sections.append("## Upstream Artifacts\n" + "\n\n".join(artifact_blocks))

    if agent_cfg.writes_artifacts:
        targets = ", ".join(f"`{name}`" for name in agent_cfg.writes_artifacts)
        sections.append(
            f"## Expected Outputs\nUse `write_artifact` to persist: {targets}."
        )

    return [
        SystemMessage(content=agent_cfg.system_prompt.strip()),
        HumanMessage(content="\n\n".join(sections)),
    ]