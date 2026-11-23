"""Base TeamAgent execution loop and config-driven AgentFactory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
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


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _extract_text_tool_calls(text: str) -> list[dict[str, Any]]:
    """Recover JSON tool calls emitted as plain text by local Ollama models."""
    decoder = json.JSONDecoder()
    calls: list[dict[str, Any]] = []
    idx = 0
    length = len(text)
    while idx < length:
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            raw_args = obj.get("arguments", obj.get("args"))
            if isinstance(raw_args, dict):
                calls.append({"name": obj["name"], "args": raw_args})
                idx = end
                continue
        idx = start + 1
    return calls


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
        max_artifact_nudges: int = 2,
    ) -> None:
        self.role = role
        self.agent_cfg = agent_cfg
        self.model = model
        self.tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.workspace_ctx = workspace_ctx
        self.max_artifact_nudges = max_artifact_nudges

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

    def _missing_artifacts(self) -> list[str]:
        return [
            name
            for name in self.agent_cfg.writes_artifacts
            if not self.workspace_ctx.resolve_artifact_path(name).is_file()
        ]

    def invoke(self, state: ProjectState) -> AgentExecutionResult:
        """Run the agent's bounded tool loop and enforce required artifact outputs."""
        upstream = load_upstream_artifacts(self.workspace_ctx, self.agent_cfg.reads_artifacts)
        messages = build_agent_messages(
            agent_cfg=self.agent_cfg,
            state=state,
            upstream_artifacts=upstream,
        )
        runnable = self._bind_runnable()
        tool_calls_made = 0
        nudges_used = 0
        final_output = ""

        for _ in range(self.agent_cfg.max_iterations):
            response = runnable.invoke(messages)
            messages.append(response)
            final_output = _message_text(response)

            native_calls: list[dict[str, Any]] = (
                [dict(c) for c in response.tool_calls]
                if isinstance(response, AIMessage) and response.tool_calls
                else []
            )
            recovered_calls: list[dict[str, Any]] = (
                [] if native_calls else _extract_text_tool_calls(final_output)
            )
            pending_calls: list[dict[str, Any]] = native_calls or recovered_calls

            if not pending_calls:
                missing = self._missing_artifacts()
                if missing and nudges_used < self.max_artifact_nudges:
                    nudges_used += 1
                    missing_list = ", ".join(f"`{name}`" for name in missing)
                    messages.append(
                        HumanMessage(
                            content=(
                                f"You did not save the required artifact(s): {missing_list}. "
                                "Do NOT just print the content in chat. Call the `write_artifact` "
                                "tool now for each missing file."
                            )
                        )
                    )
                    continue
                break

            used_native = bool(native_calls)
            recovered_results: list[str] = []
            for call in pending_calls:
                tool_calls_made += 1
                tool_name = str(call.get("name", ""))
                raw_args = call.get("args", {})
                tool_args = dict(raw_args) if isinstance(raw_args, dict) else {}
                tool_id = str(call.get("id", f"call_{tool_calls_made}"))
                tool_output = self._execute_tool(tool_name, tool_args)
                if used_native:
                    messages.append(ToolMessage(content=tool_output, tool_call_id=tool_id))
                else:
                    recovered_results.append(f"[{tool_name}] {tool_output}")

            if not used_native:
                messages.append(
                    HumanMessage(content="Tool results:\n" + "\n".join(recovered_results))
                )

        # Fallback for local models that emit valid artifact content in plain text after nudging.
        missing_after_loop = self._missing_artifacts()
        if self.max_artifact_nudges > 0 and len(missing_after_loop) == 1 and final_output.strip():
            self.workspace_ctx.ensure_artifact_dir()
            fallback_path = self.workspace_ctx.resolve_artifact_path(missing_after_loop[0])
            fallback_path.write_text(_strip_code_fence(final_output), encoding="utf-8")

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
        max_artifact_nudges: int = 2,
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
        self.max_artifact_nudges = max_artifact_nudges

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
            max_artifact_nudges=self.max_artifact_nudges,
        )