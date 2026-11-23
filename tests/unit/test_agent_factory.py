"""Unit tests for AgentFactory, bounded tool loop, nudging, and manifest tracking."""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from crewforge.agents import AgentFactory
from crewforge.config import AgentRole, load_config
from crewforge.state import ArtifactManifest, create_initial_state
from crewforge.tools import WorkspaceContext


def test_agent_executes_tool_call_and_updates_manifest(tmp_path: Path) -> None:
    cfg = load_config(env={})
    ctx = WorkspaceContext(workspace_root=tmp_path)
    state = create_initial_state(
        run_id="run-test-1",
        workspace_path=tmp_path.as_posix(),
        artifact_dir=".ai-team",
        feature_request="Add a /health endpoint",
    )

    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_artifact",
                            "args": {
                                "artifact_name": "01_prd.md",
                                "content": "# PRD\n## Objective\nExpose /health.",
                            },
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="PRD written to .ai-team/01_prd.md."),
            ]
        )
    )

    factory = AgentFactory(
        config=cfg,
        workspace_ctx=ctx,
        model_overrides={AgentRole.BUSINESS_ANALYST: fake_model},
    )
    agent = factory.create_agent(AgentRole.BUSINESS_ANALYST)
    result = agent.invoke(state)

    assert result.role == AgentRole.BUSINESS_ANALYST
    assert result.tool_calls_made == 1
    assert "01_prd.md" in result.written_artifacts
    assert (tmp_path / ".ai-team" / "01_prd.md").is_file()

    manifest_path = tmp_path / ".ai-team" / "manifest.json"
    assert manifest_path.is_file()
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert manifest.artifacts["01_prd.md"].producer == AgentRole.BUSINESS_ANALYST
    assert manifest.artifacts["01_prd.md"].revision == 1


def test_agent_cannot_invoke_disallowed_tool(tmp_path: Path) -> None:
    cfg = load_config(env={})
    ctx = WorkspaceContext(workspace_root=tmp_path)
    state = create_initial_state(
        run_id="run-test-2",
        workspace_path=tmp_path.as_posix(),
        artifact_dir=".ai-team",
        feature_request="Check tool permissions",
    )

    # Business Analyst is NOT allowed to run shell commands.
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "run_shell_command",
                            "args": {"command": "echo should-not-run"},
                            "id": "call_blocked",
                        }
                    ],
                ),
                AIMessage(content="Acknowledged, proceeding without shell access."),
            ]
        )
    )

    factory = AgentFactory(
        config=cfg,
        workspace_ctx=ctx,
        model_overrides={AgentRole.BUSINESS_ANALYST: fake_model},
        max_artifact_nudges=0,
    )
    agent = factory.create_agent(AgentRole.BUSINESS_ANALYST)
    result = agent.invoke(state)

    assert result.tool_calls_made == 1
    assert "proceeding without shell access" in result.final_output


def test_agent_nudges_when_required_artifact_missing(tmp_path: Path) -> None:
    cfg = load_config(env={})
    ctx = WorkspaceContext(workspace_root=tmp_path)
    state = create_initial_state(
        run_id="run-test-3",
        workspace_path=tmp_path.as_posix(),
        artifact_dir=".ai-team",
        feature_request="Add a /health endpoint",
    )

    # First turn prints plain text without calling write_artifact -> triggers nudge -> second turn calls tool.
    fake_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="Here is the PRD in plain text without calling the tool."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_artifact",
                            "args": {
                                "artifact_name": "01_prd.md",
                                "content": "# PRD\nSaved after nudge.",
                            },
                            "id": "call_nudge_1",
                        }
                    ],
                ),
                AIMessage(content="Saved 01_prd.md."),
            ]
        )
    )

    factory = AgentFactory(
        config=cfg,
        workspace_ctx=ctx,
        model_overrides={AgentRole.BUSINESS_ANALYST: fake_model},
    )
    agent = factory.create_agent(AgentRole.BUSINESS_ANALYST)
    result = agent.invoke(state)

    assert result.tool_calls_made == 1
    assert "01_prd.md" in result.written_artifacts
    assert (tmp_path / ".ai-team" / "01_prd.md").read_text(encoding="utf-8") == (
        "# PRD\nSaved after nudge."
    )