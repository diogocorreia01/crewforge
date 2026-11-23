"""Unit tests for OrchestratorRunner facade and Typer CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from typer.testing import CliRunner

from crewforge.cli import app
from crewforge.config import AgentRole
from crewforge.engine import OrchestratorRunner
from crewforge.state import RunStatus

_cli_runner = CliRunner()


def _fake_writer(writes: list[tuple[str, str]]) -> GenericFakeChatModel:
    tool_calls: list[ToolCall] = [
        ToolCall(
            name="write_artifact",
            args={"artifact_name": name, "content": content},
            id=f"call_{idx}",
        )
        for idx, (name, content) in enumerate(writes, start=1)
    ]
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=tool_calls),
                AIMessage(content="Done."),
            ]
        )
    )


def test_cli_init_and_doctor(tmp_path: Path) -> None:
    init_res = _cli_runner.invoke(app, ["init", "--workspace", str(tmp_path)])
    assert init_res.exit_code == 0
    assert (tmp_path / ".ai-team" / "config").is_dir()

    doc_res = _cli_runner.invoke(app, ["doctor", "--workspace", str(tmp_path)])
    assert doc_res.exit_code == 0
    assert "business_analyst" in doc_res.stdout
    assert "qwen2.5:7b" in doc_res.stdout


def test_orchestrator_runner_start_inspect_and_resume(tmp_path: Path) -> None:
    qa_json = json.dumps(
        {"passed": True, "summary": "Looks great.", "checks_run": ["pytest"], "issues": []}
    )
    models = {
        AgentRole.BUSINESS_ANALYST: _fake_writer([("01_prd.md", "# PRD")]),
        AgentRole.UI_UX_DESIGNER: _fake_writer(
            [("02_design_tokens.json", '{"ui_applicable": false}')]
        ),
        AgentRole.TECH_ARCHITECT: _fake_writer(
            [("03_api_contract.yaml", "openapi: 3.1.0"), ("04_architecture_plan.md", "# Plan")]
        ),
        AgentRole.BACKEND_DEVELOPER: _fake_writer([("05_implementation_log.md", "Done")]),
        AgentRole.FRONTEND_DEVELOPER: _fake_writer([("05_implementation_log.md", "Skipped")]),
        AgentRole.QA_REVIEWER: _fake_writer([("06_qa_report.json", qa_json)]),
    }

    runner = OrchestratorRunner(workspace_dir=tmp_path, model_overrides=models)

    paused = runner.start("Add /health endpoint", run_id="run-cli-1")
    assert paused.interrupted is True
    assert paused.status == RunStatus.AWAITING_APPROVAL

    inspected = runner.inspect_status("run-cli-1")
    assert inspected is not None
    assert inspected.status == RunStatus.AWAITING_APPROVAL

    completed = runner.resume("run-cli-1", approved=True)
    assert completed.interrupted is False
    assert completed.status == RunStatus.COMPLETED
    assert completed.qa_passed is True