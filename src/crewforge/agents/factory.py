"""Base TeamAgent execution loop and config-driven AgentFactory."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from crewforge.agents.prompts import build_agent_messages, load_upstream_artifacts
from crewforge.config.schemas import AgentConfig, AgentRole, CrewForgeConfig
from crewforge.llm.factory import build_chat_model
from crewforge.state.artifacts import ArtifactManifest
from crewforge.state.project_state import ProjectState
from crewforge.tools import (
    ToolProvider,
    WorkspaceContext,
    WorkspaceSecurityError,
    build_tool_registry,
    select_tools_for_agent,
)


@dataclass(frozen=True, slots=True)
class AgentExecutionResult:
    """Structured summary returned after a TeamAgent completes its turn."""

    role: AgentRole
    final_output: str
    written_artifacts: dict[str, str]
    tool_calls_made: int


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _update_manifest(
    *,
    ctx: WorkspaceContext,
    state: ProjectState,
    role: AgentRole,
    written_artifacts: Mapping[str, str],
) -> None:
    if not written_artifacts:
        return
    ctx.ensure_artifact_dir()
    manifest_path = ctx.resolve_artifact_path("manifest.json")
    if manifest_path.is_file():
        manifest = ArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    else:
        manifest = ArtifactManifest(
            run_id=state.get("run_id", "unknown"),
            feature_request=state.get("feature_request", ""),
            workspace_path=ctx.workspace_root.as_posix(),
        )

    for name, rel_path in written_artifacts.items():
        artifact_path = ctx.resolve_artifact_path(name)
        digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        manifest.record_artifact(
            name=name,
            relative_path=rel_path,
            producer=role,
            sha256=digest,
        )

    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


class TeamAgent:
    """Hydrated virtual team member with scoped tools and artifact contract."""

    def __init__(
        self,
        *,
        role: AgentRole,
        agent_cfg: AgentConfig,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        workspace_ctx: WorkspaceContext,
    ) -> None:
        self.role = role
        self.agent_cfg = agent_cfg
        self.model = model
        self.tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.workspace_ctx = workspace_ctx

    def _bind_runnable(self) -> Runnable[Sequence[BaseMessage], BaseMessage]:
        if not self.tools_by_name:
            return self.model
        try:
            return self.model.bind_tools(list(self.tools_by_name.values()))
        except NotImplementedError:
            # Allows simple fake/mock chat models in unit tests without tool binding.
            return self.model

    def _execute_tool(self, tool_name: str, tool_args: dict[str, object]) -> str:
        tool = self.tools_by_name.get(tool_name)
        if tool is None:
            return f"Error: tool '{tool_name}' is not allowed for role '{self.role.value}'."
        try:
            result = tool.invoke(tool_args)
            return str(result)
        except WorkspaceSecurityError as exc:
            return f"Security error: {exc}"
        except Exception as exc:  # noqa: BLE001 - return tool errors to LLM for self-correction
            return f"Tool '{tool_name}' failed: {exc}"

    def invoke(self, state: ProjectState) -> AgentExecutionResult:
        """Run the agent's bounded tool loop and record any produced artifacts."""
        upstream = load_upstream_artifacts(self.workspace_ctx, self.agent_cfg.reads_artifacts)
        messages = build_agent_messages(
            agent_cfg=self.agent_cfg,
            state=state,
            upstream_artifacts=upstream,
        )
        runnable = self._bind_runnable()
        tool_calls_made = 0
        final_output = ""

        for _ in range(self.agent_cfg.max_iterations):
            response = runnable.invoke(messages)
            messages.append(response)
            final_output = _message_text(response)

            if not isinstance(response, AIMessage) or not response.tool_calls:
                break

            for call in response.tool_calls:
                tool_calls_made += 1
                tool_name = str(call.get("name", ""))
                raw_args = call.get("args", {})
                tool_args = dict(raw_args) if isinstance(raw_args, dict) else {}
                tool_id = str(call.get("id", f"call_{tool_calls_made}"))
                tool_output = self._execute_tool(tool_name, tool_args)
                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_id))

        written: dict[str, str] = {}
        for artifact_name in self.agent_cfg.writes_artifacts:
            path = self.workspace_ctx.resolve_artifact_path(artifact_name)
            if path.is_file():
                written[artifact_name] = path.relative_to(
                    self.workspace_ctx.workspace_root
                ).as_posix()

        _update_manifest(
            ctx=self.workspace_ctx,
            state=state,
            role=self.role,
            written_artifacts=written,
        )

        return AgentExecutionResult(
            role=self.role,
            final_output=final_output,
            written_artifacts=written,
            tool_calls_made=tool_calls_made,
        )


class AgentFactory:
    """Constructs configured TeamAgent instances bound to a target workspace."""

    def __init__(
        self,
        *,
        config: CrewForgeConfig,
        workspace_ctx: WorkspaceContext,
        extra_tool_providers: Sequence[ToolProvider] = (),
        model_overrides: Mapping[AgentRole, BaseChatModel] | None = None,
    ) -> None:
        self.config = config
        self.workspace_ctx = workspace_ctx
        self.tool_registry = build_tool_registry(
            workspace_ctx,
            extra_providers=extra_tool_providers,
        )
        self.model_overrides: dict[AgentRole, BaseChatModel] = (
            dict(model_overrides) if model_overrides is not None else {}
        )

    def create_agent(self, role: AgentRole) -> TeamAgent:
        """Build a TeamAgent for the requested role with its scoped tools and LLM."""
        agent_cfg = self.config.agents.get_agent(role)
        model = self.model_overrides.get(role)
        if model is None:
            model_profile = self.config.resolve_model_for_agent(role)
            model = build_chat_model(model_profile)

        scoped_tools = select_tools_for_agent(self.tool_registry, agent_cfg.allowed_tools)
        return TeamAgent(
            role=role,
            agent_cfg=agent_cfg,
            model=model,
            tools=scoped_tools,
            workspace_ctx=self.workspace_ctx,
        )