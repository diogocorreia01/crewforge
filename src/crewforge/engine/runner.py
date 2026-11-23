"""High-level OrchestratorRunner facade over the LangGraph workflow."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from crewforge.agents import AgentFactory
from crewforge.config import AgentRole, CrewForgeConfig, load_config
from crewforge.engine.events import RunOutcome
from crewforge.graph import build_workflow_graph, open_checkpointer
from crewforge.state import RunStatus, create_initial_state
from crewforge.tools import ToolProvider, WorkspaceContext


def _extract_interrupt_payload(raw_state: Mapping[str, Any]) -> dict[str, Any] | None:
    interrupts = raw_state.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    value = getattr(first, "value", first)
    if isinstance(value, dict):
        return dict(value)
    return {"message": str(value)}


def _to_outcome(run_id: str, raw_state: Mapping[str, Any]) -> RunOutcome:
    interrupt_payload = _extract_interrupt_payload(raw_state)
    interrupted = interrupt_payload is not None
    raw_status = str(raw_state.get("status", RunStatus.RUNNING.value))
    status = RunStatus.AWAITING_APPROVAL if interrupted else RunStatus(raw_status)

    raw_artifacts = raw_state.get("artifacts", {})
    artifacts = (
        {str(k): str(v) for k, v in raw_artifacts.items()}
        if isinstance(raw_artifacts, Mapping)
        else {}
    )
    raw_errors = raw_state.get("errors", [])
    errors = [str(e) for e in raw_errors] if isinstance(raw_errors, list) else []

    return RunOutcome(
        run_id=run_id,
        status=status,
        current_stage=str(raw_state.get("current_stage", "unknown")),
        interrupted=interrupted,
        interrupt_payload=interrupt_payload,
        artifacts=artifacts,
        qa_passed=bool(raw_state.get("qa_passed", False)),
        qa_retry_count=int(raw_state.get("qa_retry_count", 0)),
        errors=errors,
    )


class OrchestratorRunner:
    """Decoupled entrypoint to start, resume, and inspect CrewForge workflow runs."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        config: CrewForgeConfig | None = None,
        config_dir: Path | None = None,
        extra_tool_providers: Sequence[ToolProvider] = (),
        model_overrides: Mapping[AgentRole, BaseChatModel] | None = None,
    ) -> None:
        self.workspace_dir = workspace_dir.resolve()
        self.config = (
            config
            if config is not None
            else load_config(config_dir=config_dir, workspace_dir=self.workspace_dir)
        )
        self.workspace_ctx = WorkspaceContext(
            workspace_root=self.workspace_dir,
            artifact_dir_name=self.config.workflow.artifact_dir_name,
        )
        self.agent_factory = AgentFactory(
            config=self.config,
            workspace_ctx=self.workspace_ctx,
            extra_tool_providers=extra_tool_providers,
            model_overrides=model_overrides,
        )

    def start(self, feature_request: str, *, run_id: str | None = None) -> RunOutcome:
        """Start a new workflow run (pauses at HITL gate if enabled)."""
        resolved_run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        initial_state = create_initial_state(
            run_id=resolved_run_id,
            workspace_path=self.workspace_ctx.workspace_root.as_posix(),
            artifact_dir=self.config.workflow.artifact_dir_name,
            feature_request=feature_request,
            max_qa_retries=self.config.workflow.max_qa_retries,
        )
        thread_cfg: RunnableConfig = {"configurable": {"thread_id": resolved_run_id}}

        with open_checkpointer(self.config.workflow.checkpointer, self.workspace_ctx) as saver:
            graph = build_workflow_graph(
                config=self.config,
                agent_factory=self.agent_factory,
                checkpointer=saver,
            )
            result = graph.invoke(initial_state, config=thread_cfg)
        return _to_outcome(resolved_run_id, result)

    def resume(
        self,
        run_id: str,
        *,
        approved: bool,
        feedback: str = "",
    ) -> RunOutcome:
        """Resume a paused HITL workflow run from its saved checkpoint."""
        thread_cfg: RunnableConfig = {"configurable": {"thread_id": run_id}}
        with open_checkpointer(self.config.workflow.checkpointer, self.workspace_ctx) as saver:
            graph = build_workflow_graph(
                config=self.config,
                agent_factory=self.agent_factory,
                checkpointer=saver,
            )
            result = graph.invoke(
                Command(resume={"approved": approved, "feedback": feedback}),
                config=thread_cfg,
            )
        return _to_outcome(run_id, result)

    def inspect_status(self, run_id: str) -> RunOutcome | None:
        """Inspect the latest persisted checkpoint state for a given `run_id`."""
        thread_cfg: RunnableConfig = {"configurable": {"thread_id": run_id}}
        with open_checkpointer(self.config.workflow.checkpointer, self.workspace_ctx) as saver:
            graph = build_workflow_graph(
                config=self.config,
                agent_factory=self.agent_factory,
                checkpointer=saver,
            )
            snapshot = graph.get_state(thread_cfg)
        if not snapshot or not snapshot.values:
            return None

        raw: dict[str, Any] = dict(snapshot.values)
        if snapshot.tasks and any(task.interrupts for task in snapshot.tasks):
            interrupts = [item for task in snapshot.tasks for item in task.interrupts]
            raw["__interrupt__"] = interrupts
        return _to_outcome(run_id, raw)