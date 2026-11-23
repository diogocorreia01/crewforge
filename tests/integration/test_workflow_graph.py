"""End-to-end integration tests for the CrewForge LangGraph workflow."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from crewforge.agents import AgentFactory
from crewforge.config import AgentRole, load_config
from crewforge.graph import build_workflow_graph, open_checkpointer
from crewforge.state import RunStatus, create_initial_state
from crewforge.tools import WorkspaceContext


def _artifact_writer_model(
    writes: list[tuple[str, str]],
    *,
    repeat_turns: int = 1,
) -> GenericFakeChatModel:
    """Create a deterministic fake LLM that calls `write_artifact` for each (name, content)."""
    messages: list[AIMessage] = []
    call_idx = 0
    for _ in range(repeat_turns):
        tool_calls = []
        for artifact_name, content in writes:
            call_idx += 1
            tool_calls.append(
                {
                    "name": "write_artifact",
                    "args": {"artifact_name": artifact_name, "content": content},
                    "id": f"call_{call_idx}",
                }
            )
        messages.append(AIMessage(content="", tool_calls=tool_calls))
        messages.append(AIMessage(content="Artifacts written."))
    return GenericFakeChatModel(messages=iter(messages))


def _qa_model(verdicts: list[bool]) -> GenericFakeChatModel:
    """Create a fake QA Reviewer LLM that writes `06_qa_report.json` for each verdict."""
    messages: list[AIMessage] = []
    for idx, passed in enumerate(verdicts, start=1):
        issues: list[dict[str, str]] = []
        if not passed:
            issues.append(
                {
                    "severity": "major",
                    "owner": "backend_developer",
                    "file_path": "src/app.py",
                    "description": "Missing /health unit test.",
                    "suggested_fix": "Add test_health_endpoint.",
                }
            )
        report: dict[str, object] = {
            "passed": passed,
            "summary": "All checks passed." if passed else "Missing health route test.",
            "checks_run": ["pytest"],
            "issues": issues,
        }
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_artifact",
                        "args": {
                            "artifact_name": "06_qa_report.json",
                            "content": json.dumps(report, indent=2),
                        },
                        "id": f"qa_call_{idx}",
                    }
                ],
            )
        )
        messages.append(AIMessage(content="QA report written."))
    return GenericFakeChatModel(messages=iter(messages))


def test_workflow_hitl_approve_and_qa_pass(tmp_path: Path) -> None:
    cfg = load_config(env={})
    ctx = WorkspaceContext(workspace_root=tmp_path)

    models = {
        AgentRole.BUSINESS_ANALYST: _artifact_writer_model([("01_prd.md", "# PRD")]),
        AgentRole.UI_UX_DESIGNER: _artifact_writer_model(
            [("02_design_tokens.json", '{"ui_applicable": false}')]
        ),
        AgentRole.TECH_ARCHITECT: _artifact_writer_model(
            [
                ("03_api_contract.yaml", "openapi: 3.1.0"),
                ("04_architecture_plan.md", "# Plan"),
            ]
        ),
        AgentRole.BACKEND_DEVELOPER: _artifact_writer_model(
            [("05_implementation_log.md", "Added /health")]
        ),
        AgentRole.FRONTEND_DEVELOPER: _artifact_writer_model(
            [("05_implementation_log.md", "No UI changes needed")]
        ),
        AgentRole.QA_REVIEWER: _qa_model([True]),
    }

    factory = AgentFactory(config=cfg, workspace_ctx=ctx, model_overrides=models)
    thread_cfg: RunnableConfig = {"configurable": {"thread_id": "run-happy-1"}}
    initial_state = create_initial_state(
        run_id="run-happy-1",
        workspace_path=tmp_path.as_posix(),
        artifact_dir=cfg.workflow.artifact_dir_name,
        feature_request="Add /health endpoint",
        max_qa_retries=cfg.workflow.max_qa_retries,
    )

    with open_checkpointer(cfg.workflow.checkpointer, ctx) as checkpointer:
        graph = build_workflow_graph(
            config=cfg, agent_factory=factory, checkpointer=checkpointer
        )

        # Phase 1: runs planning and pauses at HITL gate
        paused = graph.invoke(initial_state, config=thread_cfg)
        assert "__interrupt__" in paused
        assert (tmp_path / ".ai-team" / "01_prd.md").is_file()
        assert (tmp_path / ".ai-team" / "04_architecture_plan.md").is_file()

        # Phase 2: human approves -> Dev -> QA Pass -> Completed
        final = graph.invoke(Command(resume={"approved": True}), config=thread_cfg)
        assert final["status"] == RunStatus.COMPLETED.value
        assert final["qa_passed"] is True
        assert (tmp_path / ".ai-team" / "06_qa_report.json").is_file()
        assert (tmp_path / ".ai-team" / "checkpoints.sqlite").is_file()


def test_workflow_hitl_revision_and_qa_fix_loop(tmp_path: Path) -> None:
    cfg = load_config(env={})
    ctx = WorkspaceContext(workspace_root=tmp_path)

    models = {
        AgentRole.BUSINESS_ANALYST: _artifact_writer_model(
            [("01_prd.md", "# PRD v2")], repeat_turns=2
        ),
        AgentRole.UI_UX_DESIGNER: _artifact_writer_model(
            [("02_design_tokens.json", '{"ui_applicable": false}')], repeat_turns=2
        ),
        AgentRole.TECH_ARCHITECT: _artifact_writer_model(
            [
                ("03_api_contract.yaml", "openapi: 3.1.0"),
                ("04_architecture_plan.md", "# Plan v2"),
            ],
            repeat_turns=2,
        ),
        AgentRole.BACKEND_DEVELOPER: _artifact_writer_model(
            [("05_implementation_log.md", "Implemented & fixed")], repeat_turns=2
        ),
        AgentRole.FRONTEND_DEVELOPER: _artifact_writer_model(
            [("05_implementation_log.md", "No UI changes")], repeat_turns=2
        ),
        AgentRole.QA_REVIEWER: _qa_model([False, True]),
    }

    factory = AgentFactory(config=cfg, workspace_ctx=ctx, model_overrides=models)
    thread_cfg: RunnableConfig = {"configurable": {"thread_id": "run-loop-1"}}
    initial_state = create_initial_state(
        run_id="run-loop-1",
        workspace_path=tmp_path.as_posix(),
        artifact_dir=cfg.workflow.artifact_dir_name,
        feature_request="Add /health endpoint",
        max_qa_retries=3,
    )

    with open_checkpointer(cfg.workflow.checkpointer, ctx) as checkpointer:
        graph = build_workflow_graph(
            config=cfg, agent_factory=factory, checkpointer=checkpointer
        )

        first_pause = graph.invoke(initial_state, config=thread_cfg)
        assert "__interrupt__" in first_pause

        # Request changes -> loops back to BA -> pauses at HITL again
        second_pause = graph.invoke(
            Command(resume={"approved": False, "feedback": "Include uptime_seconds field"}),
            config=thread_cfg,
        )
        assert "__interrupt__" in second_pause

        # Approve -> Dev -> QA fails -> Dev fixes -> QA passes
        final = graph.invoke(Command(resume={"approved": True}), config=thread_cfg)
        assert final["status"] == RunStatus.COMPLETED.value
        assert final["qa_passed"] is True
        assert final["qa_retry_count"] == 1


def test_workflow_escalates_when_qa_retries_exhausted(tmp_path: Path) -> None:
    cfg = load_config(env={})
    cfg.workflow.enable_hitl_plan_gate = False
    cfg.workflow.max_qa_retries = 1
    ctx = WorkspaceContext(workspace_root=tmp_path)

    models = {
        AgentRole.BUSINESS_ANALYST: _artifact_writer_model([("01_prd.md", "# PRD")]),
        AgentRole.UI_UX_DESIGNER: _artifact_writer_model(
            [("02_design_tokens.json", '{"ui_applicable": false}')]
        ),
        AgentRole.TECH_ARCHITECT: _artifact_writer_model(
            [("03_api_contract.yaml", "openapi: 3.1.0"), ("04_architecture_plan.md", "# Plan")]
        ),
        AgentRole.BACKEND_DEVELOPER: _artifact_writer_model(
            [("05_implementation_log.md", "Impl")]
        ),
        AgentRole.FRONTEND_DEVELOPER: _artifact_writer_model(
            [("05_implementation_log.md", "Impl")]
        ),
        AgentRole.QA_REVIEWER: _qa_model([False]),
    }

    factory = AgentFactory(config=cfg, workspace_ctx=ctx, model_overrides=models)
    thread_cfg: RunnableConfig = {"configurable": {"thread_id": "run-escalate-1"}}
    initial_state = create_initial_state(
        run_id="run-escalate-1",
        workspace_path=tmp_path.as_posix(),
        artifact_dir=cfg.workflow.artifact_dir_name,
        feature_request="Add /health endpoint",
        max_qa_retries=1,
    )

    with open_checkpointer(cfg.workflow.checkpointer, ctx) as checkpointer:
        graph = build_workflow_graph(
            config=cfg, agent_factory=factory, checkpointer=checkpointer
        )
        final = graph.invoke(initial_state, config=thread_cfg)

    assert final["status"] == RunStatus.ESCALATED.value
    assert final["qa_passed"] is False
    assert final["qa_retry_count"] == 1